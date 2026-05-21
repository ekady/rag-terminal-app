from .provider import Provider
from .website import Website
from typing import Generator


class Analyzer:
    """
    Analyzes job descriptions and extracts structured information.

    This is the core logic that orchestrates:
    1. Scraping the job posting
    2. Crafting the analysis prompt
    3. Streaming the response back
    """

    system_prompt = """You are an expert career coach and job analysis specialist. 
    Your job is to analyze job postings and extract actionable information for job seekers.
    
    When analyzing a job description, extract and clearly separate:
    
    1. REQUIRED SKILLS (2-3 sentences summary, then a bullet list of 5-8 key technical and soft skills)
    2. COVER LETTER OPENING (a 2-3 sentence hook that shows you understand the role and can immediately demonstrate value)
    3. INTERVIEW PREP QUESTIONS (3 thoughtful questions you should prepare answers for based on this role's focus areas)
    
    Format your response as follows (use markdown):
    
    ## Required Skills
    [Summary paragraph]
    
    - Skill 1
    - Skill 2
    - ... (and so on)
    
    ## Cover Letter Opening
    [2-3 sentences]
    
    ## Interview Prep Questions
    1. [Question 1 - specific to the job]
    2. [Question 2 - about company/industry]
    3. [Question 3 - about your fit or growth]
    
    Be specific and insightful. Reference actual details from the job posting. Make the cover letter opening compelling and show deep understanding of the role's requirements."""

    def __init__(self, provider: Provider):
        self.provider = provider
        self.job_scrapper = Website()

    def analyze(self, job_desc: str) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": job_desc},
        ]
        return self.provider.stream_response(messages)
