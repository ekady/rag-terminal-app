from gradio import Blocks, Textbox, Markdown, Button, Theme
from .analyzer import Analyzer
from .provider import Provider
from config import Config
from .website import Website


def create_interface():
    with Blocks() as app:
        app.title = "Job Description Analyzer"
        app.description = (
            "Analyzes job descriptions and extracts structured information."
        )
        app.theme = Theme()

        Markdown("## Job Description Analyzer")
        input_text = Textbox(label="URL of the Job Description", lines=1)
        Markdown(
            "### Example URL: https://www.github.careers/careers-home/jobs/5397?lang=en-us"
        )
        submit_button = Button("Analyze")
        output_text = Markdown()

        def analyze_job_desc(url: str):
            website = Website()
            text = website.extract_text(url)
            provider = Provider(Config.LLM_MODEL)
            analyzer = Analyzer(provider)

            output = ""
            for token in analyzer.analyze(text):
                output += token
                yield output

        submit_button.click(
            fn=analyze_job_desc, inputs=[input_text], outputs=[output_text]
        )
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch()
