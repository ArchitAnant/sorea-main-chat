from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from managers.message import MessageManager
from filter import MentalHealthFilter
from config import Config
from managers.firebase_manager import FirebaseManager
from managers.summary import SummaryManager
from managers.events import EventManager
from managers.crisis import CrisisManager
from managers.helper import HelperManager
from firebase_writer import FirebaseWriter
import asyncio
import logging


class MentalHealthChatbot:
    """Main chatbot class that orchestrates the mental health conversation."""

    def __init__(self):
        logging.info("Initializing MentalHealthChatbot...")
        self.firebase_manager = FirebaseManager()
        self.writer = FirebaseWriter()
        self.config = Config()

        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            google_api_key=self.config.gemini_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        self.message_manager = MessageManager(self.firebase_manager)
        self.health_filter = MentalHealthFilter(self.config)
        self.event_manager = EventManager(self.config, self.firebase_manager)
        self.crisis_manager = CrisisManager(self.config)
        self.helper_manager = HelperManager(self.config)
        self.summary_manager = SummaryManager(self.config,self.firebase_manager.db)
        
        self.system_prompt = """You are Sorea - a caring, supportive friend who adapts your response style based on what the person needs. Your personality adjusts to match the situation:

⏰ TIME AWARENESS - VERY IMPORTANT:
        - ALWAYS acknowledge when time has passed since your last conversation
        - If they haven't talked in 1+ days, mention it: "Haven't heard from you since yesterday, how are you holding up?"
        - If it's been several days: "Man, it's been 3 days! I was worried about you. How have you been?"
        - Reference time naturally: "Last time we talked..." "Since yesterday..." "A few days ago you mentioned..."
        - If it's the same day: "Earlier today you said..." "A few hours ago..."
        - Use the time context provided to show you care and remember their timeline

        🎭 ADAPTIVE RESPONSE LEVELS:

        🟢 CASUAL/POSITIVE CONVERSATIONS (when they're sharing good news, casual chat, mild stress):
        - Be a supportive, chill friend 
        - Use encouraging language but don't overreact
        - Ask follow-up questions naturally
        - Match their energy level - if they're casual, be casual
        - Example: "That's awesome, man! How did that make you feel?" "Sounds like you're handling things well"

        🟡 MILD CONCERN (everyday stress, minor worries, feeling down but not severe):
        - Be more attentive and caring
        - Offer gentle support and encouragement  
        - Ask deeper questions but don't assume crisis
        - Provide perspective and coping suggestions
        - Example: "That sounds tough, bro. Want to talk about what's making you feel this way?"

        🟠 MODERATE DISTRESS (significant anxiety, depression symptoms, relationship issues):
        - Show more emotional investment
        - Be more direct about caring and support
        - Challenge negative thoughts gently but firmly
        - Remind them of their strengths and support system
        - Example: "Hey, I can tell this is really affecting you. You don't have to go through this alone"

        🛑 CRISIS MODE (suicidal thoughts, severe depression, immediate danger):
        - NOW you become passionate and protective
        - Fight back against harmful thoughts aggressively but lovingly
        - Remind them of people who love them (family, friends, partners)
        - Challenge their negative thoughts directly: "That's not true, bro, and you know it"
        - Be their protector: "I'm not letting you think like that, man"
        - Show urgency about their wellbeing
        🛡️ CRISIS INTERVENTION EXAMPLES:
        "Bro, STOP. Your mom said that because she's scared and stressed, not because she doesn't love you!"
        "Listen to me - you are NOT going anywhere! Your family needs you, even if they're bad at showing it right now."
        "No way, man! You think your dad sacrificed everything just to lose his son? Hell no!"

        💡 KEY PRINCIPLE: MATCH THE ENERGY AND NEED
        - Don't treat someone sharing good news like they're in crisis
        - Don't treat casual frustration like severe depression  
        - Escalate your intensity only when the situation truly calls for it
        - Be supportive without being overwhelming

        🤗 CARING CONTEXTUAL QUESTIONS (Ask these AFTER building rapport, not immediately):
        When someone seems stressed/sad/troubled, gradually ask about:
        - Basic care: "Have you been eating okay?" "How's your sleep been lately?"
        - Relationships: "Everything okay with family?" "How are things with your girlfriend/boyfriend?"
        - Life context: "What's been going on at school/work?" "Did something happen with your parents?"
        - Support system: "Do you have friends you can talk to about this?"

        ⏰ TIMING FOR DEEPER QUESTIONS:
        - NEVER ask personal questions in the first 1-2 exchanges
        - Wait until they've shared something emotional or concerning
        - Build on what they tell you naturally
        - If they mention being sad, THEN ask what happened
        - If they seem stressed, THEN explore the source

        EXAMPLE PROGRESSION:
        User: "I'm feeling really down"
        You: "I'm sorry to hear that, bro. What's been going on?"
        User: [shares more]
        You: "That sounds tough. How have you been sleeping through all this?" OR "Have you talked to anyone close to you about this?"

        Remember: You can be caring and supportive without being aggressive. Save the intense, protective energy for when someone actually needs saving."""



    # ---------------------------------------------------------------------
    async def process_conversation_async(self, email: str, message: str) -> str:
        try:
            # Fetch in parallel
            user_profile, emotion_urgency, recent_messages = await asyncio.gather(
                asyncio.to_thread(self.firebase_manager.get_user_profile, email),
                asyncio.to_thread(self.helper_manager.detect_emotion, message),
                asyncio.to_thread(self.message_manager.get_conversation, email, self.firebase_manager, None, 20)
            )

            # Last 2–3 messages
            if recent_messages:
                last_messages = [msg.user_message.content for msg in recent_messages[-3:]]
            else:
                last_messages = [message]

            # Extract emotion/urgency
            emotion, urgency_level = emotion_urgency
            user_name = user_profile.name


            if not message.startswith("[TEST]"):
                topic_filter = await asyncio.to_thread(self.health_filter.filter, last_messages)

            # Ignore non-mental-health queries
            if not topic_filter.is_mental_health_related:
                redirect = "Sorry but i can not answer to that question!!!."
                asyncio.create_task(
                    self.writer.submit(self.message_manager.add_chat_pair,
                                       email, message, redirect, emotion, urgency_level)
                )
                return redirect

            # Extract events (background)
            event_future = asyncio.create_task(
                asyncio.to_thread(self.event_manager._extract_events_with_llm, message, email)
            )

            # Crisis handling
            if urgency_level >= 5:
                crisis = self.crisis_manager.handle_crisis_situation(email, message, self.firebase_manager)
                asyncio.create_task(
                    self.writer.submit(self.message_manager.add_chat_pair,
                                       email, message, crisis.content, emotion, urgency_level)
                )
                return crisis.content

            # Add event if exists
            event = await event_future
            if event:
                asyncio.create_task(self.writer.submit(self.event_manager.add_event, email, event))

            # Normal response
            return await self._generate_response_async(
                email=email,
                message=message,
                user_name=user_name,
                emotion=emotion,
                urgency_level=urgency_level,
                recent_messages=recent_messages
            ) # type: ignore

        except Exception as e:
            logging.error(f"Error async conversation: {e}")
            return self.process_conversation_sync(email, message)


    # ---------------------------------------------------------------------
    async def _generate_response_async(self, email, message, user_name, emotion, urgency_level, recent_messages):
        try:
            enhanced_prompt = f"""
{self.system_prompt}

CONVERSATION CONTEXT:
{recent_messages}

CURRENT USER STATE:
- Emotion: {emotion}
- Urgency: {urgency_level}/5
- Name: {user_name}
"""

            messages = [SystemMessage(content=enhanced_prompt)]

            # Chat history
            if recent_messages:
                for msg_pair in recent_messages:
                    messages.append(HumanMessage(content=msg_pair.user_message.content))
                    messages.append(AIMessage(content=msg_pair.llm_message.content))

            # Add new message okay
            messages.append(HumanMessage(content=message))

            # LLM CALL
            response = await asyncio.to_thread(self.llm.invoke, messages)
            bot_message = response.content

            # Persist interaction (non-blocking for caller)
            asyncio.create_task(self.writer.submit(
                self.message_manager.add_chat_pair,
                email, message, bot_message, emotion, urgency_level
            ))

            asyncio.create_task(self.writer.submit(
                self.message_manager.add_suggestions,
                self.helper_manager,
                emotion,
                urgency_level,
                email,
                self.firebase_manager,
                self.message_manager,
                message
                
            ))
            
            return bot_message

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            raise


    # ---------------------------------------------------------------------
    def process_conversation(self, email: str, message: str) -> str:
        """Required by API + test"""
        return asyncio.run(self.process_conversation_async(email, message))


    # ---------------------------------------------------------------------
    def process_conversation_sync(self, email: str, message: str) -> str:
        """Fallback sync method"""
        try:
            user_profile = self.firebase_manager.get_user_profile(email)
            recent_messages = self.message_manager.get_conversation(email, self.firebase_manager, limit=20)
            last_messages = [msg.user_message.content for msg in recent_messages[-3:]] if recent_messages else [message]
            topic_filter = self.health_filter.filter(last_messages)
            emotion, urgency_level = self.helper_manager.detect_emotion(message)

            # TEST bypass
            if message.startswith("[TEST]"):
                return "[TEST CHAT SUCCESS]"

            if not topic_filter.is_mental_health_related:
                redirect = "Sorry but i can not answer to that question!!!."
                return redirect

            # Crisis block
            if urgency_level >= 5:
                crisis = self.crisis_manager.handle_crisis_situation(email, message, self.firebase_manager)
                return crisis.content

            enhanced_prompt = f"""
{self.system_prompt}

CONVERSATION CONTEXT:
{recent_messages}
"""

            messages = [SystemMessage(content=enhanced_prompt)]

            if recent_messages:
                for msg_pair in recent_messages:
                    messages.append(HumanMessage(content=msg_pair.user_message.content))
                    messages.append(AIMessage(content=msg_pair.llm_message.content))

            messages.append(HumanMessage(content=message))

            response = self.llm.invoke(messages)
            return response.content

            asyncio.run(self.writer.submit(
                self.message_manager.add_chat_pair,
                email, message, bot_message, emotion, urgency_level
            ))
            asyncio.run(self.writer.submit(
                self.message_manager.add_suggestions,
                self.helper_manager,
                emotion,
                urgency_level,
                email,
                self.firebase_manager,
                self.message_manager,
                message
            ))
            
            return bot_message
            
        except Exception as e:
            logging.error(f"Sync error: {e}")
            raise
