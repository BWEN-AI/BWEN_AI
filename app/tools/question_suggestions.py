from typing import List
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_relevant_questions(context: str) -> List[str]:
    """Generate relevant follow-up questions based on the conversation context."""
    
    prompt = f"""Based on this conversation context: "{context}"
    Generate 3 relevant follow-up questions that would be natural to ask next.
    Return only the questions, one per line, without numbering or prefixes.
    The questions should be diverse and explore different aspects of the topic."""
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = model.invoke(prompt)
    
    # Split the response into individual questions and clean them
    questions = [q.strip() for q in response.content.split('\n') if q.strip()]
    return questions[:3]  # Ensure we only return 3 questions 