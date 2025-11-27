#!/usr/bin/env python3
"""
Mock Chatbot Server - Simulates External Deep Research Chatbot

This Flask server implements both SSE streaming and async job polling patterns
to test the JSE Analytics Platform integration without a real chatbot.

Usage:
    # Install dependencies
    pip install flask

    # Run server
    python mock_server.py

    # Run on custom port
    python mock_server.py --port 8080

    # Test SSE streaming mode
    curl -X POST http://localhost:5000/api/v1/stream \
         -H "Content-Type: application/json" \
         -d '{"query": "What is AI?", "conversation_history": []}'

    # Test with mock client
    python mock_client.py --url http://localhost:5000/api/v1/stream --query "Test query"
"""

import argparse
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Generator
from flask import Flask, request, Response, jsonify, session, make_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'mock-chatbot-secret-key-change-in-production'

# In-memory job storage (use Redis/database in production)
jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# SSE STREAMING MODE
# ============================================================================

def generate_sse_stream(query: str, conversation_history: list) -> Generator[str, None, None]:
    """
    Generate SSE stream for real-time chatbot response.

    Yields SSE formatted strings matching the integration spec.
    """
    logger.info(f"Starting SSE stream for query: {query}")

    # Simulate processing steps
    steps = [
        {
            'event': 'progress',
            'data': {
                'step': 'initializing',
                'message': 'Initializing Deep Research analysis...',
                'progress': 10,
                'details': {'timestamp': datetime.utcnow().isoformat()}
            }
        },
        {
            'event': 'progress',
            'data': {
                'step': 'loading_documents',
                'message': 'Loading relevant documents...',
                'progress': 25,
                'details': {
                    'documents_found': 12,
                    'sources': ['Annual Report 2023', 'Q3 Financial Statement']
                }
            }
        },
        {
            'event': 'progress',
            'data': {
                'step': 'analyzing',
                'message': 'Analyzing financial data...',
                'progress': 50,
                'details': {
                    'metrics_analyzed': 15,
                    'total_metrics': 30
                }
            }
        },
        {
            'event': 'progress',
            'data': {
                'step': 'generating_response',
                'message': 'Generating comprehensive response...',
                'progress': 75,
                'details': None
            }
        },
        {
            'event': 'result',
            'data': {
                'response': f'Based on my analysis of your query "{query}", here are the key findings:\n\n'
                           f'1. The market shows strong growth indicators\n'
                           f'2. Financial metrics demonstrate stability\n'
                           f'3. Historical trends support positive outlook\n\n'
                           f'This analysis considered {len(conversation_history)} previous messages '
                           f'for context and examined 12 relevant documents.',
                'sources': [
                    {
                        'title': 'Annual Report 2023',
                        'url': 'https://example.com/reports/2023.pdf',
                        'relevance': 0.95
                    },
                    {
                        'title': 'Q3 Financial Statement',
                        'url': 'https://example.com/financials/q3.pdf',
                        'relevance': 0.87
                    }
                ],
                'metadata': {
                    'processing_time': 8.5,
                    'documents_analyzed': 12,
                    'model_version': 'mock-v1.0'
                }
            }
        },
        {
            'event': 'complete',
            'data': {
                'progress': 100,
                'message': 'Analysis complete',
                'step': 'complete'
            }
        }
    ]

    # Stream each step with delay
    for step in steps:
        event_name = step['event']
        data = json.dumps(step['data'])

        # Format as SSE
        sse_message = f"event: {event_name}\ndata: {data}\n\n"
        logger.info(f"SSE -> {event_name}: {data[:100]}...")

        yield sse_message

        # Simulate processing time (except for final complete event)
        if event_name != 'complete':
            time.sleep(1.5)

    logger.info("SSE stream completed")


@app.route('/api/v1/stream', methods=['POST'])
def stream_endpoint():
    """
    Main streaming endpoint - supports both SSE and job polling modes.

    Query Parameters:
        mode=job - Force job polling mode (default: SSE streaming)
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        conversation_history = data.get('conversation_history', [])
        auto_load_documents = data.get('auto_load_documents', True)
        memory_enabled = data.get('memory_enabled', True)

        logger.info(f"Received request - Query: {query}")
        logger.info(f"Conversation history: {len(conversation_history)} messages")
        logger.info(f"Auto-load docs: {auto_load_documents}, Memory: {memory_enabled}")

        # Set session cookie for sticky sessions
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            logger.info(f"✅ Created new session: {session_id}")
        else:
            logger.info(f"✅ Using existing session: {session_id}")

        # Determine mode (SSE vs Job)
        mode = request.args.get('mode', 'sse')

        if mode == 'job':
            # Job polling mode
            logger.info("📋 Using job polling mode")
            return create_job_response(query, conversation_history)
        else:
            # SSE streaming mode (default)
            logger.info("📡 Using SSE streaming mode")
            response = Response(
                generate_sse_stream(query, conversation_history),
                mimetype='text/event-stream'
            )
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'

            # Ensure session cookie is set
            response.set_cookie(
                'session_id',
                session_id,
                httponly=True,
                samesite='Lax'
            )

            return response

    except Exception as e:
        logger.error(f"Error in stream endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# JOB POLLING MODE
# ============================================================================

def create_job_response(query: str, conversation_history: list) -> tuple:
    """
    Create async job and return job ID for polling.
    """
    job_id = f"job_{uuid.uuid4().hex[:16]}"

    # Create job entry
    jobs[job_id] = {
        'job_id': job_id,
        'status': 'pending',
        'query': query,
        'conversation_history': conversation_history,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'latest_progress': {
            'step': 'pending',
            'message': 'Job queued for processing',
            'progress': 0
        },
        'result': None,
        'error': None,
        # Store session ID to validate sticky sessions
        'session_id': session.get('session_id')
    }

    logger.info(f"✅ Created job: {job_id}")
    logger.info(f"📝 Job session: {jobs[job_id]['session_id']}")

    response_data = {
        'job_id': job_id,
        'status': 'pending',
        'polling_url': f'/jobs/{job_id}'
    }

    # Set session cookie
    response = make_response(jsonify(response_data))
    response.set_cookie(
        'session_id',
        session.get('session_id'),
        httponly=True,
        samesite='Lax'
    )

    return response


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """
    Poll job status endpoint.

    Critical: Validates session cookie for sticky session routing.
    """
    logger.info(f"🔄 Polling job: {job_id}")

    # Validate session cookie
    client_session = session.get('session_id')
    logger.info(f"🔑 Client session: {client_session}")

    if job_id not in jobs:
        logger.error(f"❌ Job not found: {job_id}")
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    # Validate sticky session (optional in mock, but demonstrates the pattern)
    if job['session_id'] != client_session:
        logger.warning(f"⚠️ Session mismatch! Job session: {job['session_id']}, Client: {client_session}")
        # In production, this might return 401 or route to correct worker

    # Simulate job progression
    current_status = job['status']

    if current_status == 'pending':
        # Move to running
        job['status'] = 'running'
        job['latest_progress'] = {
            'step': 'loading_documents',
            'message': 'Loading relevant documents...',
            'progress': 25,
            'details': {'documents_found': 8}
        }
        logger.info(f"📊 Job {job_id}: pending -> running")

    elif current_status == 'running':
        # Advance progress
        current_progress = job['latest_progress']['progress']

        if current_progress < 50:
            job['latest_progress'] = {
                'step': 'analyzing',
                'message': 'Analyzing financial data...',
                'progress': 50,
                'details': {'metrics_analyzed': 10, 'total_metrics': 20}
            }
        elif current_progress < 75:
            job['latest_progress'] = {
                'step': 'generating_response',
                'message': 'Generating response...',
                'progress': 75,
                'details': None
            }
        else:
            # Complete the job
            job['status'] = 'succeeded'
            job['latest_progress'] = {
                'step': 'complete',
                'message': 'Analysis complete',
                'progress': 100
            }
            job['result'] = {
                'response': f'Based on my analysis of your query "{job["query"]}", here are the findings:\n\n'
                           f'1. Market shows positive indicators\n'
                           f'2. Financial stability confirmed\n'
                           f'3. Growth trajectory is sustainable\n\n'
                           f'Analysis completed using job polling pattern.',
                'sources': [
                    {
                        'title': 'Market Analysis Report',
                        'url': 'https://example.com/analysis.pdf',
                        'relevance': 0.92
                    }
                ],
                'metadata': {
                    'processing_time': 6.2,
                    'documents_analyzed': 8,
                    'job_id': job_id
                }
            }
            logger.info(f"✅ Job {job_id}: completed successfully")

    job['updated_at'] = datetime.utcnow().isoformat()

    # Return current job status
    response_data = {
        'job_id': job['job_id'],
        'status': job['status'],
        'latest_progress': job['latest_progress'],
        'result': job['result'],
        'error': job['error'],
        'created_at': job['created_at'],
        'updated_at': job['updated_at']
    }

    logger.info(f"📊 Job {job_id} status: {job['status']} ({job['latest_progress']['progress']}%)")

    return jsonify(response_data)


# ============================================================================
# ERROR SIMULATION ENDPOINTS
# ============================================================================

@app.route('/api/v1/stream/error', methods=['POST'])
def error_stream():
    """Test endpoint that always returns an error."""
    logger.info("Simulating error response")

    def generate_error():
        yield "event: progress\n"
        yield 'data: {"step": "initializing", "message": "Starting...", "progress": 10}\n\n'
        time.sleep(1)

        yield "event: error\n"
        yield 'data: {"error": "Simulated processing error: Document retrieval failed"}\n\n'

    response = Response(generate_error(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/api/v1/stream/timeout', methods=['POST'])
def timeout_stream():
    """Test endpoint that simulates a timeout (very slow response)."""
    logger.info("Simulating slow/timeout response")

    def generate_slow():
        yield "event: progress\n"
        yield 'data: {"step": "processing", "message": "This will take a while...", "progress": 5}\n\n'

        # Wait longer than typical timeout (2+ minutes)
        time.sleep(150)

        yield "event: complete\n"
        yield 'data: {"progress": 100, "message": "Finally done!"}\n\n'

    response = Response(generate_slow(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.route('/jobs/<job_id>/fail', methods=['POST'])
def fail_job(job_id: str):
    """Manually fail a job (for testing error handling)."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    jobs[job_id]['status'] = 'failed'
    jobs[job_id]['error'] = 'Manually failed for testing purposes'
    jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()

    logger.info(f"❌ Manually failed job: {job_id}")

    return jsonify({'message': 'Job marked as failed', 'job_id': job_id})


# ============================================================================
# HEALTH & DEBUG ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'active_jobs': len(jobs)
    })


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs (for debugging)."""
    return jsonify({
        'jobs': [
            {
                'job_id': job_id,
                'status': job_data['status'],
                'created_at': job_data['created_at'],
                'progress': job_data['latest_progress']['progress']
            }
            for job_id, job_data in jobs.items()
        ]
    })


@app.route('/jobs/<job_id>/delete', methods=['DELETE'])
def delete_job(job_id: str):
    """Delete a job (cleanup for testing)."""
    if job_id in jobs:
        del jobs[job_id]
        logger.info(f"🗑️  Deleted job: {job_id}")
        return jsonify({'message': 'Job deleted', 'job_id': job_id})
    return jsonify({'error': 'Job not found'}), 404


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Mock Chatbot Server - Test integration with JSE Analytics Platform'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run server on (default: 5000)'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode'
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MOCK CHATBOT SERVER                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Server running at: http://{args.host}:{args.port}

ENDPOINTS:
  SSE Streaming:
    POST /api/v1/stream
         {"query": "Your question", "conversation_history": []}

  Job Polling:
    POST /api/v1/stream?mode=job
         {"query": "Your question"}
    GET  /jobs/<job_id>

  Error Testing:
    POST /api/v1/stream/error     - Simulates processing error
    POST /api/v1/stream/timeout   - Simulates timeout
    POST /jobs/<job_id>/fail      - Manually fail a job

  Debug:
    GET  /health                   - Health check
    GET  /jobs                     - List all jobs
    DELETE /jobs/<job_id>/delete   - Delete job

TEST WITH MOCK CLIENT:
  python mock_client.py --url http://{args.host}:{args.port}/api/v1/stream --query "Test"
  python mock_client.py --url http://{args.host}:{args.port}/api/v1/stream?mode=job --query "Test" --expect-job

Press Ctrl+C to stop
╔══════════════════════════════════════════════════════════════════════════════╗
    """)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
