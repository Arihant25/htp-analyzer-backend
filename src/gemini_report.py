"""
Gemini Report Generator for HTP Analysis
Generates psychological interpretation reports using Gemini API with RAG context.
"""

import os
from typing import Optional

from google import genai
from google.genai import types


class GeminiReportGenerator:
    """Generate psychological reports using Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini report generator.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini model name for text generation
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set in environment or passed as parameter"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def generate_psychological_interpretation(
        self,
        analysis_features: dict,
        rag_context: str,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate psychological interpretation using Gemini with RAG context.

        Args:
            analysis_features: Dictionary containing analysis results
            rag_context: Retrieved context from knowledge base
            max_tokens: Maximum tokens for response

        Returns:
            Generated psychological interpretation
        """
        # Build prompt with analysis and context
        prompt = self._build_prompt(analysis_features, rag_context)

        # Generate response using Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_output_tokens=max_tokens,
            ),
        )

        return response.text

    def _format_characteristics(self, characteristics: dict) -> str:
        """Format characteristics dictionary for prompt display."""
        if not characteristics:
            return "  No data available"
        
        formatted_lines = []
        for key, value in characteristics.items():
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            formatted_lines.append(f"  - {key.replace('_', ' ').title()}: {value}")
        
        return '\n'.join(formatted_lines) if formatted_lines else "  No data available"

    def _build_prompt(self, analysis_features: dict, rag_context: str) -> str:
        """Build comprehensive prompt for Gemini."""
        prompt = f"""You are an expert psychologist specializing in House-Tree-Person (HTP) projective drawing analysis. 

Based on the following HTP drawing analysis results and reference material from the HTP interpretation guide, provide a professional psychological interpretation.

## DRAWING ANALYSIS RESULTS:

**Detected Features:** {', '.join(analysis_features.get('detected_features', []))}
**Missing Features:** {', '.join(analysis_features.get('missing_features', []))}

**House Characteristics:**
- Size Category: {analysis_features.get('house_size_category', 'unknown')}
- Area Ratio: {analysis_features.get('house_area_ratio', 0):.3f}
- Placement: {' and '.join(analysis_features.get('house_placement', []))}

**Feature Details:**
- Door Present: {analysis_features.get('door_present', False)}
- Window Count: {analysis_features.get('window_count', 0)}
- Chimney Present: {analysis_features.get('chimney_present', False)}

**Detailed Size Analysis:**

*Door Characteristics:*
{self._format_characteristics(analysis_features.get('door_characteristics', {}))}

*Window Characteristics:*
{self._format_characteristics(analysis_features.get('window_characteristics', {}))}

*Chimney Characteristics:*
{self._format_characteristics(analysis_features.get('chimney_characteristics', {}))}

*Roof Characteristics:*
{self._format_characteristics(analysis_features.get('roof_characteristics', {}))}

*Wall Characteristics:*
{self._format_characteristics(analysis_features.get('wall_characteristics', {}))}

**Preliminary Indicators:**
- Risk Factors: {', '.join(analysis_features.get('risk_factors', []))}
- Positive Indicators: {', '.join(analysis_features.get('positive_indicators', []))}

## REFERENCE MATERIAL FROM HTP GUIDE:

{rag_context}

## INSTRUCTIONS:

Based on the analysis results and reference material above, provide a comprehensive psychological interpretation that includes:

1. **Overall Assessment**: Summarize the key psychological indicators observed in the drawing
2. **Size and Proportion Analysis**: Interpret the psychological significance of the sizes and proportions of detected features (door, windows, chimney, roof, etc.) relative to the house and each other
3. **Specific Feature Interpretations**: Explain the psychological significance of present and missing features
4. **Emotional and Behavioral Indicators**: Describe what the drawing suggests about the subject's emotional state and behavioral tendencies
5. **Clinical Considerations**: Highlight any risk factors or areas that may warrant further clinical attention, particularly those indicated by disproportionate sizing
6. **Positive Aspects**: Note any positive indicators of psychological wellbeing

Keep the interpretation:
- Professional and evidence-based
- Grounded in the reference material provided
- Balanced between concerns and strengths
- Appropriate for clinical documentation
- Approximately 300-500 words

**IMPORTANT**: This is an automated analysis tool and should be used only as a preliminary assessment. Final interpretations should be made by qualified mental health professionals who can consider the complete clinical context.

Generate the psychological interpretation now:"""

        return prompt


def main():
    """Test Gemini report generation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return

    # Test with sample analysis
    test_analysis = {
        "detected_features": ["house", "roof", "wall"],
        "missing_features": ["door", "window", "chimney"],
        "house_size_category": "small",
        "house_area_ratio": 0.08,
        "house_placement": ["left", "low"],
        "door_present": False,
        "window_count": 0,
        "chimney_present": False,
        "risk_factors": ["withdrawal", "social difficulties", "isolation", "insecurity"],
        "positive_indicators": [],
    }

    test_context = """Missing doors in house drawings often indicate difficulty in social relationships and 
    connecting with others. This can suggest feelings of insecurity or fearfulness about engagement 
    with the environment. Small house size (less than 10% of page area) is associated with feelings 
    of inadequacy and rejection of home life. Low placement on the page can indicate insecurity and 
    low self-esteem. Absence of windows suggests a guarded or suspicious nature."""

    print("🤖 Generating psychological interpretation with Gemini...")
    generator = GeminiReportGenerator(api_key=api_key)

    interpretation = generator.generate_psychological_interpretation(
        test_analysis, test_context
    )

    print("\n" + "=" * 80)
    print("PSYCHOLOGICAL INTERPRETATION:")
    print("=" * 80)
    print(interpretation)
    print("=" * 80)


if __name__ == "__main__":
    main()
