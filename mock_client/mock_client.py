#!/usr/bin/env python3
"""
Mock Platform Client - Simulates JSE Analytics Platform's Chatbot Integration

This script replicates how the platform interacts with the external chatbot service.
Use it to test your chatbot server implementation without needing the full platform.

Usage:
    # Test SSE streaming mode
    python mock_client.py --url https://your-chatbot.com/api/v1/stream --query "What is the stock market?"

    # Test job polling mode
    python mock_client.py --url https://your-chatbot.com/api/v1/stream --query "Analyze financial reports" --expect-job

    # Test with conversation history
    python mock_client.py --url https://your-chatbot.com/api/v1/stream --query "What about XYZ?" --history history.json

Requirements:
    pip install requests
"""

import argparse
import json
import logging
import sys
import time
from typing import Generator, Tuple, Optional, Dict, Any
from urllib.parse import urljoin

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_POLL_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_POLL_INTERVAL = 2.0  # seconds


class MockPlatformClient:
    """
    Simulates the JSE Analytics Platform's chatbot client.

    Replicates the behavior of:
    - analytics/services/streaming_chat_service.py
    - frontend/src/services/streamingService.ts
    """

    def __init__(self, base_url: str, poll_interval: float = DEFAULT_POLL_INTERVAL):
        """
        Initialize the mock client.

        Args:
            base_url: Chatbot service URL (e.g., https://chatbot.com/api/v1/stream)
            poll_interval: Seconds between polling attempts (default: 2.0)
        """
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/stream'):
            self.base_url += '/stream'

        self.poll_interval = poll_interval
        self.session = requests.Session()
        logger.info(f"Initialized mock client with URL: {self.base_url}")

    def send_query(
        self,
        query: str,
        conversation_history: Optional[list] = None,
        auto_load_documents: bool = True,
        memory_enabled: bool = True
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Send query to chatbot and stream responses.

        Yields (event, data) tuples matching platform behavior.

        Args:
            query: User question
            conversation_history: Previous Q&A pairs
            auto_load_documents: Enable document auto-loading
            memory_enabled: Enable conversation memory

        Yields:
            Tuples of (event_name, data_json_string)
        """
        payload = {
            "query": query,
            "conversation_history": conversation_history or [],
            "auto_load_documents": auto_load_documents,
            "memory_enabled": memory_enabled,
        }

        headers = {"Content-Type": "application/json"}

        logger.info(f"Sending query: {query}")
        logger.info(f"Conversation history length: {len(conversation_history or [])}")

        try:
            with self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=120
            ) as resp:
                resp.raise_for_status()
                logger.info(f"Response status: {resp.status_code}")

                # Check for session cookies (critical for sticky sessions)
                if resp.cookies:
                    logger.info(f"✅ Cookies received: {dict(resp.cookies)}")
                    logger.info(f"✅ Session cookies: {dict(self.session.cookies)}")
                else:
                    logger.warning("⚠️ NO COOKIES received - sticky sessions may not work!")

                # Determine response type
                content_type = resp.headers.get('content-type', '').lower()
                logger.info(f"Response content-type: {content_type}")

                if 'text/event-stream' in content_type:
                    logger.info("📡 Detected SSE streaming response")
                    yield from self._process_sse_stream(resp)
                elif 'application/json' in content_type:
                    logger.info("📋 Detected JSON job response")
                    job_payload = resp.json()
                    logger.info(f"Job payload: {job_payload}")

                    # Extract base URL for polling
                    base_url = self.base_url.rsplit('/stream', 1)[0]
                    yield from self._poll_job_status(job_payload, base_url)
                else:
                    logger.error(f"❌ Unexpected content-type: {content_type}")
                    yield 'error', json.dumps({"error": f"Unexpected response type: {content_type}"})

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request error: {e}")
            yield 'error', json.dumps({"error": f"Failed to connect to chatbot: {str(e)}"})
        except Exception as e:
            logger.exception(f"❌ Unexpected error: {e}")
            yield 'error', json.dumps({"error": str(e)})

    def _process_sse_stream(self, resp: requests.Response) -> Generator[Tuple[str, str], None, None]:
        """
        Process SSE stream from chatbot.

        Replicates: streaming_chat_service.py:_process_sse_stream

        Yields:
            (event, data) tuples
        """
        try:
            buffer = ''
            event = ''

            for chunk in resp.iter_content(chunk_size=1024):
                if not chunk:
                    continue

                text = chunk.decode('utf-8')
                logger.debug(f"Raw chunk: {repr(text)}")
                buffer += text

                lines = buffer.split('\n')
                buffer = lines.pop() if lines else ''

                for line in lines:
                    logger.debug(f"Processing line: {repr(line)}")

                    if line.strip() == '':
                        continue

                    if line.startswith('event: '):
                        event = line[7:]
                        logger.info(f"📨 Event: {event}")
                    elif line.startswith('data: '):
                        data = line[6:]
                        logger.info(f"📦 Data: {data[:100]}...")

                        # Normalize event name
                        normalized_event = event.strip() if event and event.strip() != '' else 'message'
                        yield normalized_event, data

                        # Reset event for next iteration
                        event = ''

        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"❌ Chunked encoding error: {e}")
            yield 'error', json.dumps({"error": "Stream ended prematurely"})

    def _poll_job_status(
        self,
        job_payload: Dict[str, Any],
        base_url: str
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Poll job status and yield SSE-compatible events.

        Replicates: streaming_chat_service.py:_poll_job_status

        Args:
            job_payload: Initial job response with job_id
            base_url: Base URL for constructing polling endpoint

        Yields:
            (event, data) tuples
        """
        job_id = job_payload.get('job_id')
        polling_url = job_payload.get('polling_url')

        if not job_id:
            logger.error("❌ Job payload missing job_id")
            yield 'error', json.dumps({"error": "Invalid job response: missing job_id"})
            return

        # Construct polling URL
        if polling_url:
            if polling_url.startswith('http'):
                status_url = polling_url
            else:
                status_url = urljoin(base_url + '/', polling_url)
        else:
            status_url = urljoin(base_url + '/', f"jobs/{job_id}")

        logger.info(f"🔄 Polling job at: {status_url}")
        logger.info(f"📝 Session cookies: {dict(self.session.cookies)}")

        max_polls = int(MAX_POLL_TIMEOUT_SECONDS / self.poll_interval)
        poll_count = 0

        while poll_count < max_polls:
            try:
                logger.info(f"🔄 Poll attempt {poll_count + 1}/{max_polls}")

                # Log cookies being sent
                cookies_to_send = dict(self.session.cookies)
                logger.info(f"🔑 Sending cookies: {cookies_to_send}")

                resp = self.session.get(status_url, timeout=30)
                logger.info(f"📥 Poll response status: {resp.status_code}")

                resp.raise_for_status()

                job_status = resp.json()
                logger.info(f"📊 Job status: {json.dumps(job_status, indent=2)}")

                # Transform job status to SSE events
                for event, data in self._transform_job_status_to_events(job_status):
                    yield event, data

                # Check if job is complete
                status = job_status.get('status', '').lower()
                if status in ['succeeded', 'failed', 'completed']:
                    logger.info(f"✅ Job finished with status: {status}")
                    break

                # Wait before next poll
                time.sleep(self.poll_interval)
                poll_count += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Polling error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"❌ Response: {e.response.status_code} - {e.response.text[:200]}")
                yield 'error', json.dumps({"error": f"Job polling error: {str(e)}"})
                break
            except Exception as e:
                logger.exception(f"❌ Unexpected polling error: {e}")
                yield 'error', json.dumps({"error": f"Job polling error: {str(e)}"})
                break

        if poll_count >= max_polls:
            logger.error("❌ Job polling timeout")
            yield 'error', json.dumps({"error": "Job polling timeout exceeded"})

    def _transform_job_status_to_events(
        self,
        job_status: Dict[str, Any]
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Transform job status to SSE events.

        Replicates: streaming_chat_service.py:_transform_job_status_to_events

        Args:
            job_status: Job status response from chatbot

        Yields:
            (event, data) tuples
        """
        status = job_status.get('status', '').lower()

        # Extract progress information
        latest_progress = job_status.get('latest_progress')
        if not latest_progress and job_status.get('progress'):
            progress_list = job_status.get('progress', [])
            if progress_list and isinstance(progress_list, list):
                latest_progress = progress_list[-1]

        # Yield progress event
        if latest_progress:
            if isinstance(latest_progress, dict):
                progress_data = {
                    'step': latest_progress.get('step', 'processing'),
                    'message': latest_progress.get('message', ''),
                    'progress': latest_progress.get('progress', 0),
                    'details': latest_progress.get('details')
                }
                logger.info(f"📈 Progress: {progress_data['message']} ({progress_data['progress']}%)")
                yield 'progress', json.dumps(progress_data)
            else:
                logger.warning(f"⚠️ Malformed progress data: {type(latest_progress).__name__}")

        # Handle terminal states
        if status == 'succeeded' or status == 'completed':
            result = job_status.get('result', {})
            if result:
                logger.info(f"✅ Result received: {str(result)[:200]}...")
                yield 'result', json.dumps(result)
            else:
                logger.warning("⚠️ Job succeeded but no result found")
        elif status == 'failed':
            error_msg = job_status.get('error', 'Job failed')
            logger.error(f"❌ Job failed: {error_msg}")
            yield 'error', json.dumps({"error": error_msg})


def load_conversation_history(filepath: str) -> list:
    """Load conversation history from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}")
        return []


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Mock Platform Client - Test chatbot integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test SSE streaming
  python mock_client.py --url https://chatbot.com/api/v1/stream --query "What is AI?"

  # Test job polling
  python mock_client.py --url https://chatbot.com/api/v1/stream --query "Deep analysis" --expect-job

  # With conversation history
  python mock_client.py --url https://chatbot.com/api/v1/stream --query "Continue" --history history.json

  # Verbose logging
  python mock_client.py --url https://chatbot.com/api/v1/stream --query "Test" --verbose
        """
    )

    parser.add_argument(
        '--url',
        required=True,
        help='Chatbot service URL (e.g., https://chatbot.com/api/v1/stream)'
    )
    parser.add_argument(
        '--query',
        required=True,
        help='Query to send to chatbot'
    )
    parser.add_argument(
        '--history',
        help='Path to JSON file with conversation history'
    )
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f'Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL})'
    )
    parser.add_argument(
        '--no-auto-load',
        action='store_true',
        help='Disable automatic document loading'
    )
    parser.add_argument(
        '--no-memory',
        action='store_true',
        help='Disable conversation memory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    parser.add_argument(
        '--expect-job',
        action='store_true',
        help='Expect job polling response (helps with validation)'
    )
    parser.add_argument(
        '--save-output',
        help='Save events to JSON file'
    )

    args = parser.parse_args()

    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load conversation history
    conversation_history = []
    if args.history:
        conversation_history = load_conversation_history(args.history)
        logger.info(f"Loaded {len(conversation_history)} messages from history")

    # Initialize client
    client = MockPlatformClient(args.url, args.poll_interval)

    # Send query and process events
    events_received = []
    result_found = False
    error_found = False

    print("\n" + "="*80)
    print("STREAMING EVENTS")
    print("="*80 + "\n")

    try:
        for event, data in client.send_query(
            query=args.query,
            conversation_history=conversation_history,
            auto_load_documents=not args.no_auto_load,
            memory_enabled=not args.no_memory
        ):
            events_received.append({'event': event, 'data': data})

            # Pretty print events
            try:
                data_obj = json.loads(data)
                print(f"[{event.upper()}]")
                print(json.dumps(data_obj, indent=2))
                print()

                # Track result/error
                if event == 'result':
                    result_found = True
                    if 'response' in data_obj:
                        print("="*80)
                        print("FINAL RESPONSE:")
                        print("="*80)
                        print(data_obj['response'])
                        print("="*80 + "\n")
                elif event == 'error':
                    error_found = True
            except json.JSONDecodeError:
                print(f"[{event.upper()}] {data}\n")

        # Save output if requested
        if args.save_output:
            with open(args.save_output, 'w') as f:
                json.dump(events_received, f, indent=2)
            logger.info(f"Saved {len(events_received)} events to {args.save_output}")

        # Summary
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total events received: {len(events_received)}")
        print(f"Result found: {result_found}")
        print(f"Error occurred: {error_found}")

        # Validation warnings
        if args.expect_job and not any(e['event'] == 'progress' for e in events_received):
            logger.warning("⚠️ Expected job polling but no progress events received")

        if not result_found and not error_found:
            logger.warning("⚠️ No result or error event received - stream may be incomplete")

        sys.exit(0 if result_found else 1)

    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()
