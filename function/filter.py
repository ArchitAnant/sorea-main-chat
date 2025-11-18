from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from data import MentalHealthTopicFilter


class MentalHealthFilter:
    """Filter to ensure conversations stay focused on mental health topics."""
    
    def __init__(self, config):
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.gemini_api_key,
            temperature=0.3 
        )
    
    def filter(self, last_messages: list[str]) -> MentalHealthTopicFilter:
        """
        Analyze last 2-3 user messages for mental health relevance with confidence and reason.
        """
       

        system_prompt = """ You are a mental health topic classifier for a therapeutic chatbot named MyBro.
        
        Your task:
        - Read the LAST FEW user messages (2-3 messages) (IF PRESENTED)
        - Determine whether the FINAL message is mental-health related.
        
        A message is mental-health related IF:
        1) It directly discusses emotions, stress, anxiety, depression, relationships,
           pressure, self-care, healing, personal struggles, or psychological well-being.
        OR
        2) The message connects to previous messages that were mental-health related,
           even if the final message alone is unclear.
        
        Respond ONLY in this exact format:
        MENTAL_HEALTH: YES/NO
        CONFIDENCE: <0.1-1.0>
        REASON: <short explanation>
        """

        # Format conversation
        conversation_text = "\n".join(
            [f"Message {i+1}: {msg}" for i, msg in enumerate(last_messages)]
        )

        final_message = last_messages[-1]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Here are the last user messages:\n{conversation_text}\n\n"
                    f"The FINAL message is:\n\"{final_message}\"\n\n"
                    "Decide ONLY based on whether the FINAL message is mental health related."
                )
            )
        ]

        response = self.llm.invoke(messages)

        # Safe extraction
        response_text = (response.content or "").strip()  # type: ignore

        lines = response_text.split("\n")
        is_mental_health = None
        confidence = None
        reason = None

        for line in lines:
            if line.startswith("MENTAL_HEALTH:"):
                is_mental_health = "YES" in line.upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.1, min(1.0, confidence))
                except:
                    confidence = 0.1
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return MentalHealthTopicFilter(
            is_mental_health_related=bool(is_mental_health),
            confidence_score=confidence if confidence is not None else 0.1,
            reason=reason if reason else "LLM did not provide a reason."
        )
