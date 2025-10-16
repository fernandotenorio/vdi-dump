import os
import logging
import io
from typing import List, Union
from google import genai
from google.genai import types
from PDFSplitter import split_pdf_by_page_count
from src.models.job import PartOcrResult


def _deident_string(s):
    s = '\n'.join([m.lstrip() for m in s.split('\n')])
    return s.strip()

def _base_prompt():
    instruction = '''
    Você é um sistema especialista em extração de dados. Sua tarefa é analisar as páginas de um documento e extrair o textos e possíveis tabelas.
    Siga rigorosamente as seguintes regras para o conteúdo que você extrair:

    1.  **Tabelas**: Devem ser formatadas como tabelas HTML.
    2.  **Texto Geral**: Preserve todos os parágrafos, cabeçalhos, rodapés e números de página. Evite retornar linhas em branco desnecessárias.
    4. **Para gráficos e diagramas (Regra Especial de Análise):** Não extraia pontos individuais dos gráficos, em vez disso, faça uma **análise concisa**. Sua análise deve incluir:'''

    instruction = _deident_string(instruction)

    instruction+= '''
        *   **Identificação**: O tipo de gráfico, seu título e a descrição dos eixos.
        *   **Tendência Principal**: Descreva o padrão geral dos dados.
        *   **Pontos-Chave**: Mencione apenas os valores mais significativos, como o ponto inicial, final, o valor máximo e o mínimo.
        *   **Insight Principal**: Se possível, resuma a principal conclusão que o gráfico transmite em uma frase.'''

    instruction+= '\n5. **Para imagens e fotografias (Regra Especial de Descrição):** Forneça uma **descrição breve e objetiva** da imagem. Sua descrição deve incluir:'
    instruction+= '''
        *   **Identificação**: O que a imagem retrata.
        *   **Contexto**: O propósito da imagem no documento.
        *   Evite interpretações subjetivas ou descrições excessivamente detalhadas.
    '''

    instruction+= '\nRetorne apenas o resultado da extração e nada mais.'
    return instruction

class GeminiAsyncOCR:
    """
    Wrapper class for Gemini PDF OCR using the Google GenAI client.
    Ensures required environment variables are set, validates configuration,
    and initializes the GenAI client.
    """

    _MAX_OUTPUT_TOKENS: int = 65536
    _MAX_PAGES_PER_SPLIT: int = 7
    _REQUIRED_ENV_VARS = (
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "GEMINI_MODEL_ID",
        "PAGES_PER_SPLIT",
    )

    _SAFETY_SETTINGS: List[types.SafetySetting] = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    def __init__(self) -> None:
        """Initialize the GeminiPDFOCR client and configuration."""
        self._validate_env_vars()
        self.client = genai.Client().aio
        self._model_id: str = os.getenv("GEMINI_MODEL_ID")
        self._pages_per_split: int = self._get_pages_per_split_env_var()
        self._prompt: str = _base_prompt()

    @property
    def model_id(self) -> str:
        """Read-only access to the Gemini model ID."""
        return self._model_id

    @property
    def pages_per_split(self) -> int:
        """Read-only access to the number of pages per split."""
        return self._pages_per_split

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def _validate_env_vars(self) -> None:
        """
        Ensure all required environment variables are set.
        Accepts alternative variable names in tuples.
        """
        for var in self._REQUIRED_ENV_VARS:
            if isinstance(var, tuple):
                if not any(os.getenv(v) for v in var):
                    logging.error(f"Missing required env var: one of {var} must be set")
                    raise ValueError(f"At least one of {var} must be set")
            else:
                if not os.getenv(var):
                    logging.error(f"Missing required env var: {var}")
                    raise ValueError(f"Environment variable '{var}' must be set")

    def _get_pages_per_split_env_var(self) -> int:
        """
        Retrieve and validate the PAGES_PER_SPLIT environment variable.
        Returns:
            int: Number of pages per split.
        Raises:
            ValueError: If invalid or exceeds MAX_PAGES_PER_SPLIT.
        """
        try:
            pages_per_split = int(os.getenv("PAGES_PER_SPLIT", "0"))
            if not (0 < pages_per_split <= self._MAX_PAGES_PER_SPLIT):
                raise ValueError(
                    f"PAGES_PER_SPLIT must be between 1 and {self._MAX_PAGES_PER_SPLIT}"
                )
            return pages_per_split
        except ValueError as e:
            logging.error("Invalid PAGES_PER_SPLIT value: %s", e)
            raise


   async def run(
        self, file: io.BytesIO, output_prefix: str = "part"
    ) -> Tuple[List[PartOcrResult], int] | None:
        """
        Run OCR on a PDF, splitting it and processing each part with the Gemini model.

        Returns:
            A tuple containing:
            - A list of PartOcrResult objects.
            - The total number of tokens consumed.
            Returns None if the initial splitting fails.
        """
        try:
            parts = split_pdf_by_page_count(
                file,
                pages_per_part=self.pages_per_split,
                output_prefix=output_prefix
            )
        except Exception as e:
            logging.error("Failed to split PDF: %s", e, exc_info=True)
            return None

        config = types.GenerateContentConfig(
            max_output_tokens=self._MAX_OUTPUT_TOKENS,
            temperature=0.1,
            candidate_count=1,
            safety_settings=self.__class__._SAFETY_SETTINGS
        )
        
        part_results: List[PartOcrResult] = []
        total_tokens = 0
        part_counter = 1

        for filename, pdf_stream in parts:
            pdf_stream.seek(0)

            try:
                response = await self.client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        types.Part.from_bytes(data=pdf_stream.read(), mime_type="application/pdf"),
                        self.prompt
                    ],
                    config=config
                )
            except Exception as e:
                logging.error(f"Failed to generate OCR for part {part_counter} ({filename}): {e}", exc_info=True)
                part_counter += 1
                continue

            if not response.candidates:
                logging.error(f"OCR Failed: LLM response has no candidates for part {part_counter} ({filename})")
                part_counter += 1
                continue
            
            # Use a helper to process the candidate and create the result object
            result_obj = self._process_candidate(response, part_counter)
            if result_obj:
                part_results.append(result_obj)
                total_tokens += result_obj.token_count
            
            part_counter += 1
        
        return part_results, total_tokens

    def _process_candidate(self, response: types.GenerateContentResponse, part_number: int) -> PartOcrResult | None:
        """Helper to extract data from a response candidate."""
        text_result = ""
        candidate = response.candidates[0]
        
        # Handle cases where the model did not finish properly
        if candidate.finish_reason != genai.types.FinishReason.STOP:
            logging.warning(
                "Model finished with reason %s for part %s. Attempting to retrieve partial text.",
                candidate.finish_reason.name,
                part_number
            )
            try:
                # Safely access the text part
                if candidate.content and candidate.content.parts:
                    text_result = candidate.content.parts[0].text or ""
            except Exception as e:
                logging.error(f"Failed to get partial text from part {part_number}: {e}", exc_info=True)
                return None
        else:
            # Normal successful case
            text_result = candidate.text
        
        tokens_used = response.usage_metadata.total_token_count

        return PartOcrResult(
            part_number=part_number,
            text=text_result,
            char_count=len(text_result),
            token_count=tokens_used
        )


    async def close(self):
        await self.client.aclose()


