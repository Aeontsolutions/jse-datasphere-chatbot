# Load Testing Guide for JSE Datasphere Chatbot API

This guide provides comprehensive load testing solutions for your FastAPI endpoints:

- `/chat` - Regular chat endpoint
- `/chat/stream` - Streaming chat endpoint  
- `/fast_chat_v2` - Financial data query endpoint
- `/fast_chat_v2/stream` - Streaming financial data endpoint

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r load_test_requirements.txt

# Install Apache Bench (optional, for additional stress testing)
# On macOS:
brew install httpd

# On Ubuntu/Debian:
sudo apt-get install apache2-utils
```

### 2. Start Your FastAPI Server

```bash
cd fastapi_app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run Load Tests

Choose from the following options:

## 📊 Option 1: Python Async Load Tester (Recommended)

The `load_test.py` script provides comprehensive async load testing with detailed metrics.

### Basic Usage

```bash
# Test all endpoints
python load_test.py

# Test specific endpoint
python load_test.py --endpoint chat
python load_test.py --endpoint fast_chat_v2
python load_test.py --endpoint chat/stream
python load_test.py --endpoint fast_chat_v2/stream

# Customize test parameters
python load_test.py --users 50 --duration 120 --url http://localhost:8000
```

### Command Line Options

- `--endpoint`: Endpoint to test (`chat`, `chat/stream`, `fast_chat_v2`, `fast_chat_v2/stream`, `all`)
- `--users`: Number of concurrent users (default: 10)
- `--duration`: Test duration in seconds (default: 60)
- `--url`: API base URL (default: http://localhost:8000)

### Example Output

```
Starting load test for: all
Base URL: http://localhost:8000
Concurrent users: 10
Duration: 60 seconds
Start time: 2024-01-15 14:30:00

Testing chat...
Testing chat/stream...
Testing fast_chat_v2...
Testing fast_chat_v2/stream...

================================================================================
LOAD TEST RESULTS SUMMARY
================================================================================

/CHAT ENDPOINT:
--------------------------------------------------
Total Requests: 150
Successful: 148
Failed: 2
Success Rate: 98.67%
Average Response Time: 1.234s
Median Response Time: 1.156s
Min Response Time: 0.892s
Max Response Time: 2.456s
95th Percentile: 2.123s
```

## 🕷️ Option 2: Locust Load Testing (Advanced)

Locust provides a web-based UI for load testing with real-time charts and metrics.

### Installation

```bash
pip install locust
```

### Usage

```bash
# Start Locust with the provided locustfile.py
locust -f locustfile.py --host=http://localhost:8000

# Open http://localhost:8089 in your browser
```

### Features

- **ChatUser**: Simulates users interacting with chat endpoints
- **FinancialUser**: Simulates users querying financial data
- **MixedUser**: Simulates users using both endpoint types
- Real-time metrics and charts
- Configurable user behavior patterns
- Web-based UI for monitoring

### Locust UI Configuration

1. **Number of users**: Set the total number of simulated users
2. **Spawn rate**: Users per second to start
3. **Host**: Your API base URL
4. **Start swarming**: Begin the load test

## 🔥 Option 3: Shell Script Stress Testing (Quick Tests)

The `stress_test.sh` script provides quick stress testing using Apache Bench and curl.

### Basic Usage

```bash
# Test chat endpoint
./stress_test.sh --endpoint chat --requests 1000 --concurrent 50

# Test financial endpoint
./stress_test.sh --endpoint fast_chat_v2 --requests 500 --concurrent 25

# Test streaming endpoints
./stress_test.sh --endpoint chat/stream --requests 200
./stress_test.sh --endpoint fast_chat_v2/stream --requests 200

# Custom API URL
./stress_test.sh --endpoint chat --url http://api.example.com
```

### Command Line Options

- `--endpoint`: Endpoint to test
- `--requests`: Total number of requests
- `--concurrent`: Number of concurrent requests
- `--url`: API base URL

### What It Tests

- **Apache Bench**: High-performance load testing (if available)
- **Curl Stress Test**: Sequential request testing with detailed metrics
- **Streaming Test**: Special handling for streaming endpoints

## 📈 Understanding Test Results

### Key Metrics

1. **Response Time**
   - Average: Mean response time across all requests
   - Median: Middle value (less affected by outliers)
   - 95th Percentile: 95% of requests completed within this time
   - Min/Max: Range of response times

2. **Throughput**
   - Requests per Second (RPS): How many requests your API can handle
   - Success Rate: Percentage of successful requests

3. **Concurrency**
   - How well your API handles multiple simultaneous requests
   - Connection pooling and resource management

### Performance Benchmarks

| Endpoint | Expected RPS | Expected Response Time | Notes |
|----------|--------------|----------------------|-------|
| `/chat` | 50-200 | 1-3s | Document loading + AI processing |
| `/chat/stream` | 30-100 | 2-5s | Streaming with real-time updates |
| `/fast_chat_v2` | 100-300 | 0.5-2s | Database queries + AI formatting |
| `/fast_chat_v2/stream` | 50-150 | 1-4s | Streaming financial data |

*Note: Actual performance depends on your hardware, database performance, and AI model response times.*

## 🛠️ Customization

### Modifying Test Queries

Edit the query arrays in the load testing files:

```python
# In load_test.py or locustfile.py
self.chat_queries = [
    "Your custom query here",
    "Another test query",
    # ... more queries
]
```

### Adjusting Test Parameters

```python
# In load_test.py
wait_time = between(1, 3)  # Wait 1-3 seconds between requests

# In locustfile.py
@task(3)  # Weight: 3 - more frequent requests
```

### Custom Test Scenarios

Create new user classes in Locust for specific use cases:

```python
class PowerUser(HttpUser):
    """Simulates a power user with rapid requests"""
    wait_time = between(0.1, 0.5)
    
    @task(5)
    def rapid_queries(self):
        # Your custom test logic
        pass
```

## 🔍 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure your FastAPI server is running
   - Check the API URL in your test configuration

2. **High Error Rates**
   - Monitor server logs for errors
   - Check database connections and AI service availability
   - Reduce concurrent users if server is overwhelmed

3. **Slow Response Times**
   - Check database query performance
   - Monitor AI model response times
   - Verify S3 document loading performance

4. **Memory Issues**
   - Monitor server memory usage during tests
   - Check for memory leaks in conversation history
   - Consider implementing request rate limiting

### Debug Mode

Enable debug logging in your FastAPI app:

```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

### Monitoring During Tests

```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor API logs
tail -f fastapi_app/app/logs/app.log
```

## 📊 Advanced Testing Scenarios

### 1. Endurance Testing

Test your API over extended periods:

```bash
# 1-hour test with moderate load
python load_test.py --users 20 --duration 3600
```

### 2. Spike Testing

Test how your API handles sudden traffic spikes:

```bash
# Start with 10 users, spike to 100, then back to 10
# Use Locust UI for dynamic user scaling
```

### 3. Stress Testing

Find your API's breaking point:

```bash
# Gradually increase load until failure
python load_test.py --users 10 --duration 300
python load_test.py --users 50 --duration 300
python load_test.py --users 100 --duration 300
```

### 4. Database Performance Testing

Focus on database-heavy operations:

```bash
# Test financial queries that hit BigQuery
python load_test.py --endpoint fast_chat_v2 --users 30 --duration 600
```

## 🚀 Production Considerations

### 1. Rate Limiting

Implement rate limiting to protect your API:

```python
# In your FastAPI app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, ...):
    # Your endpoint logic
    pass
```

### 2. Load Balancing

For high-traffic scenarios, consider:

- Multiple API instances behind a load balancer
- Database connection pooling
- Redis for session management
- CDN for static content

### 3. Monitoring and Alerting

Set up monitoring for:

- Response time thresholds
- Error rate spikes
- Resource utilization
- Database performance

## 📝 Test Report Template

After running tests, document your findings:

```markdown
# Load Test Report - [Date]

## Test Configuration
- Endpoints tested: [List endpoints]
- Concurrent users: [Number]
- Test duration: [Duration]
- API version: [Version]

## Results Summary
- Total requests: [Number]
- Success rate: [Percentage]
- Average response time: [Time]
- Peak RPS: [Number]

## Performance Analysis
- [Analysis of results]
- [Bottlenecks identified]
- [Recommendations]

## Next Steps
- [Action items]
- [Follow-up tests]
- [Optimization plans]
```

## 🤝 Contributing

To improve the load testing tools:

1. Add new test scenarios
2. Improve error handling
3. Add more detailed metrics
4. Create additional user behavior patterns
5. Optimize test performance

## 📚 Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [FastAPI Performance](https://fastapi.tiangolo.com/tutorial/performance/)
- [Async Python](https://docs.python.org/3/library/asyncio.html)
- [Load Testing Best Practices](https://k6.io/blog/load-testing-best-practices/)

---

Happy load testing! 🚀

