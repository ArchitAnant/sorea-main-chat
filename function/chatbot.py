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

        self.system_prompt = """You are MyBro - a caring supportive chatbot..."""


    # ---------------------------------------------------------------------
    async def process_conversation_async(self, email: str, message: str) -> str:
        """Async conversation processing."""
        try:
            # fetch all required data
            (user_profile, emotion_urgency, recent_messages) = await asyncio.gather(
                asyncio.to_thread(self.firebase_manager.get_user_profile, email),
                asyncio.to_thread(self.helper_manager.detect_emotion, message),
                asyncio.to_thread(self.message_manager.get_conversation, email, self.firebase_manager, None, 20)
            )

            # Extract last 2–3 user messages
            if recent_messages:
                last_messages = [
                    msg.user_message.content
                    for msg in recent_messages[-3:]
                ]
            else:
                last_messages = [message]  # fallback (test case)

            # Filter with correct list
            topic_filter = await asyncio.to_thread(
                self.health_filter.filter,
                last_messages
            )

            emotion, urgency_level = emotion_urgency
            user_name = user_profile.name

            # For automated test - ignore filter
            if '[TEST]' not in message:
                if not topic_filter.is_mental_health_related:
                    redirect_response = "Sorry but i can not answer to that question!!!."
                    asyncio.create_task(self.writer.submit(
                        self.message_manager.add_chat_pair,
                        email, message, redirect_response, emotion, urgency_level
                    ))
                    return redirect_response

            # extract events
            event_future = asyncio.create_task(asyncio.to_thread(
                self.event_manager._extract_events_with_llm, message, email
            ))

            if urgency_level >= 5:
                crisis_response = self.crisis_manager.handle_crisis_situation(email, message, self.firebase_manager)

                asyncio.create_task(self.writer.submit(
                    self.message_manager.add_chat_pair,
                    email, message, crisis_response.content, emotion, urgency_level
                ))

                return crisis_response.content

            event = await event_future
            if event:
                asyncio.create_task(self.writer.submit(self.event_manager.add_event, email, event))

            # normal response
            bot_message = await self._generate_response_async(
                email=email,
                message=message,
                user_name=user_name,
                emotion=emotion,
                urgency_level=urgency_level,
                recent_messages=recent_messages
            )

            return bot_message

        except Exception as e:
            logging.error(f"Error in async conversation processing: {e}")
            return self.process_conversation_sync(email, message)


    # ---------------------------------------------------------------------
    async def _generate_response_async(self, email, message, user_name, emotion, urgency_level, recent_messages):
        """Generate final LLM response."""
        try:
            enhanced_prompt = f"""
            {self.system_prompt}
            CONVERSATION CONTEXT:
            {recent_messages}
            """

            # build messages
            messages = [SystemMessage(content=enhanced_prompt)]

            if recent_messages:
                for msg_pair in recent_messages:
                    messages.append(HumanMessage(content=msg_pair.user_message.content))
                    messages.append(AIMessage(content=msg_pair.llm_message.content))

            messages.append(HumanMessage(content=message))

            response = await asyncio.to_thread(self.llm.invoke, messages)
            bot_message = response.content

            asyncio.create_task(self.writer.submit(
                self.message_manager.add_chat_pair,
                email, message, bot_message, emotion, urgency_level
            ))
            return bot_message

        except Exception as e:
            logging.error(f"Error generating async response: {e}")
            raise


    # ---------------------------------------------------------------------
    def process_conversation(self, email, message):
        return asyncio.run(self.process_conversation_async(email, message))


    # ---------------------------------------------------------------------
    def process_conversation_sync(self, email: str, message: str) -> str:
        """Fallback sync processor."""
        try:
            user_profile = self.firebase_manager.get_user_profile(email)
            user_name = user_profile.name

            recent_messages = self.message_manager.get_conversation(email, self.firebase_manager, limit=20)

            # extract last msgs
            if recent_messages:
                last_messages = [
                    msg.user_message.content
                    for msg in recent_messages[-3:]
                ]
            else:
                last_messages = [message]

            topic_filter = self.health_filter.filter(last_messages)
            emotion, urgency_level = self.helper_manager.detect_emotion(message)

            if not topic_filter.is_mental_health_related:
                redirect_response = "Sorry but i can not answer to that question!!!."
                asyncio.run(self.writer.submit(
                    self.message_manager.add_chat_pair,
                    email, message, redirect_response, emotion, urgency_level
                ))
                return redirect_response

            # normal response logic
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
            logging.error(f"Error in sync conversation processing: {e}")
            raise
