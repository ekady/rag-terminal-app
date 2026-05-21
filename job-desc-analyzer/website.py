# Reference: notebooks/website_summarize.ipynb

import requests
from bs4 import BeautifulSoup


class Website:
    """Fetches a website and extracts clean visible text."""

    def _fetch_and_parse(self, url: str) -> str:
        """Download and parse the HTML."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup

    def _remove_boilerplate(self, soup: BeautifulSoup):
        """Remove scripts, styles, and navigation elements."""
        for tag_name in ["script", "style", "meta", "link"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        boilerplate_identifiers = [
            "nav",
            "navbar",
            "navigation",
            "menu",
            "header",
            "footer",
            "sidebar",
            "ads",
            "advertisement",
            "cookie",
            "modal",
            "popup",
        ]

        for element in soup.find_all(True):
            if getattr(element, "attrs", None) is None:
                continue
                
            element_classes = element.get("class", [])
            if isinstance(element_classes, str):
                element_classes = [element_classes]
            elif element_classes is None:
                element_classes = []
                
            element_id = element.get("id", "")
            if element_id is None:
                element_id = ""

            if any(
                keyword in " ".join(element_classes).lower()
                for keyword in boilerplate_identifiers
            ):
                element.decompose()
            elif any(
                keyword in element_id.lower() for keyword in boilerplate_identifiers
            ):
                element.decompose()
        return soup

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract and clean visible text."""
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def extract_text(self, url: str) -> str:
        """Return the website's visible text content."""
        soup = self._fetch_and_parse(url)
        soup = self._remove_boilerplate(soup)
        return self._extract_text(soup)
