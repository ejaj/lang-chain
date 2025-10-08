import asyncio
from rstream import Producer

STREAM_NAME = "hello-stream"
# 5GB
STREAM_RETENTION = 5000000000


async def send():
    async with Producer(
            host="localhost",
            username="guest",
            password="guest",
    ) as producer:
        await producer.create_stream(
            STREAM_NAME, exists_ok=True, arguments={"max-length-bytes": STREAM_RETENTION}
        )

        await producer.send(stream=STREAM_NAME, message=b"Hello, World!")
        print(" [x] Hello, World! message sent")

        # optional pause so the async send finishes before exit
        input(" [x] Press Enter to close the producer...")


if __name__ == "__main__":
    asyncio.run(send())
