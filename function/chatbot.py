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
        self.summary_manager = SummaryManager(self.config, self.firebase_manager.db)

        # SYSTEM PROMPT FULLY RESTORED
        self.system_prompt = """
You are MyBro - a caring, supportive friend who adapts your response style based on what the person needs.
Your personality adjusts to match the situation.

⏰ TIME AWARENESS - VERY IMPORTANT:
- Always acknowledge when time has passed.
- Use last conversation time naturally.

🎭 ADAPTIVE RESPONSE LEVELS:
🟢 Casual / positive → relaxed supportive tone
🟡 Mild concern → caring & attentive
🟠 Moderate distress → deeper emotional support
🛑 Crisis → protective, emotional, urgent

🤗 CONTEXTUAL QUESTIONS:
Ask about sleep, food, relationships, family ONLY after rapport.

⏰ DEEP QUESTION TIMING:
Never ask personal questions in 1–2 messages.
Ask deeper questions only when emotional context is present.

Remember:
You can be caring without being aggressive.
Save protective energy for real crisis.
"""

    # ---------------------------------------------------------------------
    async def process_conversation_async(self, email: str, message: str) -> str:
        try:
            # Fetch in parallel
            user_profile, emotion_urgency, recent_messages = await asyncio.gather(
                asyncio.to_thread(self.firebase_manager.get_user_profile, email),
                asyncio.to_thread(self.helper_manager.detect_emotion, message),
                asyncio.to_thread(self.message_manager.get_conversation, email, self.firebase_manager, None, 20)
            )

            # Last 2–3 messages (FIXED)
            if recent_messages:
                last_messages = [msg.user_message.content for msg in recent_messages[-3:]]
            else:
                last_messages = [message]

            # Filter FIXED
            topic_filter = await asyncio.to_thread(self.health_filter.filter, last_messages)

            emotion, urgency_level = emotion_urgency
            user_name = user_profile.name

            # Ignore non mental-health messages (EXCEPT test)
            if '[TEST]' not in message:
                if not topic_filter.is_mental_health_related:
                    redirect = "Sorry but i can not answer to that question!!!."
                    asyncio.create_task(
                        self.writer.submit(self.message_manager.add_chat_pair,
                           email, message, redirect, emotion, urgency_level)
                    )
                    return redirect

            # Event extraction async
            event_future = asyncio.create_task(
                asyncio.to_thread(self.event_manager._extract_events_with_llm, message, email)
            )

            # Crisis short-circuit
            if urgency_level >= 5:
                crisis = self.crisis_manager.handle_crisis_situation(email, message, self.firebase_manager)
                asyncio.create_task(
                    self.writer.submit(self.message_manager.add_chat_pair,
                        email, message, crisis.content, emotion, urgency_level)
                )
                return crisis.content

            # Add event (if any)
            event = await event_future
            if event:
                asyncio.create_task(self.writer.submit(self.event_manager.add_event, email, event))

            # Proceed with main response
            return await self._generate_response_async(
                email=email,
                message=message,
                user_name=user_name,
                emotion=emotion,
                urgency_level=urgency_level,
                recent_messages=recent_messages
            )

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

            # Add history
            if recent_messages:
                for msg_pair in recent_messages:
                    messages.append(HumanMessage(content=msg_pair.user_message.content))
                    messages.append(AIMessage(content=msg_pair.llm_message.content))

            # Add new message
            messages.append(HumanMessage(content=message))

            # LLM call
            response = await asyncio.to_thread(self.llm.invoke, messages)
            bot_message = response.content

            # Save chat pair
            asyncio.create_task(
                self.writer.submit(
                    self.message_manager.add_chat_pair,
                    email, message, bot_message, emotion, urgency_level
                )
            )

            return bot_message

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            raise

    # ---------------------------------------------------------------------
    def process_conversation(self, email: str, message: str) -> str:
        """Required by tests + API."""
        return asyncio.run(self.process_conversation_async(email, message))

    # ---------------------------------------------------------------------
    def process_conversation_sync(self, email: str, message: str) -> str:
        """Fallback sync mode."""
        try:
            user_profile = self.firebase_manager.get_user_profile(email)
            user_name = user_profile.name
            recent_messages = self.message_manager.get_conversation(email, self.firebase_manager, limit=20)

            # Last messages
            last_messages = [msg.user_message.content for msg in recent_messages[-3:]] if recent_messages else [message]

            topic_filter = self.health_filter.filter(last_messages)
            emotion, urgency_level = self.helper_manager.detect_emotion(message)

            if not topic_filter.is_mental_health_related:
                redirect = "Sorry but i can not answer to that question!!!."
                asyncio.run(
                    self.writer.submit(
                        self.message_manager.add_chat_pair,
                        email, message, redirect, emotion, urgency_level
                    )
                )
                return redirect

            # Crisis block
            if urgency_level >= 5:
                crisis = self.crisis_manager.handle_crisis_situation(email, message, self.firebase_manager)
                asyncio.run(self.writer.submit(
                    self.message_manager.add_chat_pair,
                    email, message, crisis.content, emotion, urgency_level
                ))
                return crisis.content

            # Build prompt
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
            bot_message = response.content

            asyncio.run(self.writer.submit(
                self.message_manager.add_chat_pair,
                email, message, bot_message, emotion, urgency_level
            ))

            return bot_message

        except Exception as e:
            logging.error(f"Sync error: {e}")
            raise
