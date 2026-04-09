import os
import json
from pydantic import ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from .models import StudentProfile

load_dotenv()
load_dotenv(dotenv_path="../.env")

api_key = os.environ.get("GOOGLE_API_KEY")

# Temperature 0.0 is better for extraction/interviews to keep AI focused
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=api_key, 
    temperature=0.0 
)

class ChatProfiler:
    def __init__(self):
        self.messages = [
            SystemMessage(content="""You are a Placement IQ Assistant. Ask about:
1. Academic_Score (CGPA)
2. DSA_Skill (1-10)
3. Soft_Skills (1-10)
4. Project_Quality (1-10)
5. Experience (Internships/Work)
6. Tech Stack (Languages/Tools)
Ask ONE short question at a time. Review history to avoid repeating. 
End with [PROFILE_COMPLETE] ONLY when all 6 points are answered.""")
        ]
        
    async def astream_chat(self, user_input: str):
        if not user_input or user_input.strip() == "":
            return

        # 1. Add user message
        self.messages.append(HumanMessage(content=user_input))
            
        # 2. FIXED: Create a FLAT list of messages
        # We take the system prompt and add it to the recent history slice
        if len(self.messages) > 7:
            messages_to_send = [self.messages] + self.messages[-6:]
        else:
            messages_to_send = self.messages
        
        full_content = ""
        try:
            async for chunk in llm.astream(messages_to_send):
                content = chunk.content
                full_content += content
                yield content
        except Exception as e:
            yield f"[ERROR] AI Link Down: {str(e)}"
            return

        # 3. Save AI response and filter empty ones
        if full_content.strip():
            self.messages.append(AIMessage(content=full_content))
            
    def is_complete(self):
        if not self.messages or len(self.messages) < 2: return False
        return "[PROFILE_COMPLETE]" in self.messages[-1].content

    def extract_profile(self) -> StudentProfile:
        # Filter messages to ensure only valid text is sent for extraction
        history = "\n".join([f"{type(m).__name__}: {m.content}" for m in self.messages if m.content])
        prompt = f"""
        Map this conversation to JSON. Output ONLY raw JSON.
        Fields: Academic_Score (float), DSA_Skill (float), Soft_Skills (float), 
        Project_Quality (float), Experience_Score (float), 
        OpenSource_Value (float), Tech_Stack_Score (float), tech_stack (list of strings).
        
        Conversation:
        {history}
        """
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            # Cleaner extraction logic
            json_str = resp.content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json").split("```")
            elif "```" in json_str:
                json_str = json_str.split("```").split("```")
            
            data = json.loads(json_str.strip())
            
            return StudentProfile(
                academic_score=float(data.get("Academic_Score", 0)),
                dsa_skill=float(data.get("DSA_Skill", 0)),
                project_quality=float(data.get("Project_Quality", 0)),
                experience_score=float(data.get("Experience_Score", 0)),
                opensource_value=float(data.get("OpenSource_Value", 0)),
                soft_skills=float(data.get("Soft_Skills", 0)),
                tech_stack_score=float(data.get("Tech_Stack_Score", 0)),
                tech_stack=data.get("tech_stack", [])
            )
        except Exception as e:
            print(f"Extraction failed: {e}")
            return StudentProfile()