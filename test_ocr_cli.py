from GeminiAsyncOCR import *
from dotenv import load_dotenv
import asyncio
load_dotenv()


async def run():
    with open("docs/ata.pdf", "rb") as f:
        fs = io.BytesIO(f.read())

    cli = GeminiAsyncOCR()
    ocr = await cli.run(fs)
    await cli.close()

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(ocr)


if __name__ == "__main__":
    asyncio.run(run())