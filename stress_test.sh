#!/bin/bash

# Stress Testing Script for JSE Datasphere Chatbot API
# This script provides quick stress testing using Apache Bench and curl

set -e

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
ENDPOINT="${ENDPOINT:-chat}"
REQUESTS="${REQUESTS:-1000}"
CONCURRENT="${CONCURRENT:-50}"
DURATION="${DURATION:-60}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check API health
check_api_health() {
    print_status "Checking API health..."
    
    if curl -s -f "${API_BASE_URL}/docs" > /dev/null; then
        print_success "API is accessible at ${API_BASE_URL}"
    else
        print_error "API is not accessible at ${API_BASE_URL}"
        print_error "Make sure your FastAPI server is running"
        exit 1
    fi
}

# Function to run Apache Bench test
run_ab_test() {
    local endpoint=$1
    local requests=$2
    local concurrent=$3
    
    print_status "Running Apache Bench test for ${endpoint}"
    print_status "Requests: ${requests}, Concurrent: ${concurrent}"
    
    # Create test payload
    local payload_file="/tmp/test_payload_${endpoint}.json"
    
    case $endpoint in
        "chat")
            cat > "$payload_file" << EOF
{
    "query": "What are the key financial metrics for JSE companies?",
    "conversation_history": [],
    "auto_load_documents": true,
    "memory_enabled": true
}
EOF
            ;;
        "fast_chat_v2")
            cat > "$payload_file" << EOF
{
    "query": "Show me MDS revenue for 2024",
    "conversation_history": [],
    "memory_enabled": true
}
EOF
            ;;
        *)
            print_error "Unknown endpoint: ${endpoint}"
            return 1
            ;;
    esac
    
    # Run Apache Bench
    if command_exists ab; then
        print_status "Running Apache Bench..."
        ab -n "$requests" -c "$concurrent" \
           -p "$payload_file" \
           -T "application/json" \
           -v 2 \
           "${API_BASE_URL}/${endpoint}" 2>&1 | tee "ab_results_${endpoint}.txt"
        
        print_success "Apache Bench test completed. Results saved to ab_results_${endpoint}.txt"
    else
        print_warning "Apache Bench (ab) not found. Install it to run this test."
        print_status "On macOS: brew install httpd"
        print_status "On Ubuntu/Debian: sudo apt-get install apache2-utils"
    fi
    
    # Cleanup
    rm -f "$payload_file"
}

# Function to run curl-based stress test
run_curl_stress_test() {
    local endpoint=$1
    local requests=$2
    local concurrent=$3
    
    print_status "Running curl-based stress test for ${endpoint}"
    print_status "Requests: ${requests}, Concurrent: ${concurrent}"
    
    # Create test payload
    local payload_file="/tmp/test_payload_${endpoint}.json"
    
    case $endpoint in
        "chat")
            cat > "$payload_file" << EOF
{
    "query": "What are the key financial metrics for JSE companies?",
    "conversation_history": [],
    "auto_load_documents": true,
    "memory_enabled": true
}
EOF
            ;;
        "fast_chat_v2")
            cat > "$payload_file" << EOF
{
    "query": "Show me MDS revenue for 2024",
    "conversation_history": [],
    "memory_enabled": true
}
EOF
            ;;
        *)
            print_error "Unknown endpoint: ${endpoint}"
            return 1
            ;;
    esac
    
    # Run curl stress test
    print_status "Starting curl stress test..."
    
    local start_time=$(date +%s)
    local success_count=0
    local error_count=0
    local total_time=0
    
    for i in $(seq 1 "$requests"); do
        local request_start=$(date +%s.%N)
        
        if curl -s -f \
            -X POST \
            -H "Content-Type: application/json" \
            -d "@$payload_file" \
            "${API_BASE_URL}/${endpoint}" > /dev/null 2>&1; then
            ((success_count++))
        else
            ((error_count++))
        fi
        
        local request_end=$(date +%s.%N)
        local request_time=$(echo "$request_end - $request_start" | bc -l 2>/dev/null || echo "0")
        total_time=$(echo "$total_time + $request_time" | bc -l 2>/dev/null || echo "0")
        
        # Progress indicator
        if ((i % 100 == 0)); then
            print_status "Progress: ${i}/${requests} requests completed"
        fi
        
        # Small delay to avoid overwhelming
        sleep 0.01
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Calculate statistics
    local success_rate=$(echo "scale=2; $success_count * 100 / $requests" | bc -l 2>/dev/null || echo "0")
    local avg_time=$(echo "scale=3; $total_time / $requests" | bc -l 2>/dev/null || echo "0")
    local rps=$(echo "scale=2; $requests / $duration" | bc -l 2>/dev/null || echo "0")
    
    # Print results
    echo
    print_success "Curl stress test completed!"
    echo "=================================="
    echo "Endpoint: ${endpoint}"
    echo "Total Requests: ${requests}"
    echo "Duration: ${duration}s"
    echo "Successful: ${success_count}"
    echo "Failed: ${error_count}"
    echo "Success Rate: ${success_rate}%"
    echo "Average Response Time: ${avg_time}s"
    echo "Requests per Second: ${rps}"
    
    # Save results
    cat > "curl_results_${endpoint}.txt" << EOF
Curl Stress Test Results for ${endpoint}
========================================
Total Requests: ${requests}
Duration: ${duration}s
Successful: ${success_count}
Failed: ${error_count}
Success Rate: ${success_rate}%
Average Response Time: ${avg_time}s
Requests per Second: ${rps}
EOF
    
    print_success "Results saved to curl_results_${endpoint}.txt"
    
    # Cleanup
    rm -f "$payload_file"
}

# Function to run streaming endpoint test
run_streaming_test() {
    local endpoint=$1
    local requests=$2
    
    print_status "Running streaming endpoint test for ${endpoint}"
    print_status "Requests: ${requests}"
    
    # Create test payload
    local payload_file="/tmp/test_payload_${endpoint}.json"
    
    case $endpoint in
        "chat/stream")
            cat > "$payload_file" << EOF
{
    "query": "What are the key financial metrics for JSE companies?",
    "conversation_history": [],
    "auto_load_documents": true,
    "memory_enabled": true
}
EOF
            ;;
        "fast_chat_v2/stream")
            cat > "$payload_file" << EOF
{
    "query": "Show me MDS revenue for 2024",
    "conversation_history": [],
    "memory_enabled": true
}
EOF
            ;;
        *)
            print_error "Unknown streaming endpoint: ${endpoint}"
            return 1
            ;;
    esac
    
    # Run streaming test
    print_status "Starting streaming test..."
    
    local start_time=$(date +%s)
    local success_count=0
    local error_count=0
    local total_time=0
    
    for i in $(seq 1 "$requests"); do
        local request_start=$(date +%s.%N)
        
        if curl -s -f \
            -X POST \
            -H "Content-Type: application/json" \
            -d "@$payload_file" \
            "${API_BASE_URL}/${endpoint}" > /dev/null 2>&1; then
            ((success_count++))
        else
            ((error_count++))
        fi
        
        local request_end=$(date +%s.%N)
        local request_time=$(echo "$request_end - $request_start" | bc -l 2>/dev/null || echo "0")
        total_time=$(echo "$total_time + $request_time" | bc -l 2>/dev/null || echo "0")
        
        # Progress indicator
        if ((i % 50 == 0)); then
            print_status "Progress: ${i}/${requests} streaming requests completed"
        fi
        
        # Small delay for streaming endpoints
        sleep 0.05
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Calculate statistics
    local success_rate=$(echo "scale=2; $success_count * 100 / $requests" | bc -l 2>/dev/null || echo "0")
    local avg_time=$(echo "scale=3; $total_time / $requests" | bc -l 2>/dev/null || echo "0")
    local rps=$(echo "scale=2; $requests / $duration" | bc -l 2>/dev/null || echo "0")
    
    # Print results
    echo
    print_success "Streaming test completed!"
    echo "============================="
    echo "Endpoint: ${endpoint}"
    echo "Total Requests: ${requests}"
    echo "Duration: ${duration}s"
    echo "Successful: ${success_count}"
    echo "Failed: ${error_count}"
    echo "Success Rate: ${success_rate}%"
    echo "Average Response Time: ${avg_time}s"
    echo "Requests per Second: ${rps}"
    
    # Save results
    cat > "streaming_results_${endpoint//\//_}.txt" << EOF
Streaming Test Results for ${endpoint}
=====================================
Total Requests: ${requests}
Duration: ${duration}s
Successful: ${success_count}
Failed: ${error_count}
Success Rate: ${success_rate}%
Average Response Time: ${avg_time}s
Requests per Second: ${rps}
EOF
    
    print_success "Results saved to streaming_results_${endpoint//\//_}.txt"
    
    # Cleanup
    rm -f "$payload_file"
}

# Main function
main() {
    echo "🚀 JSE Datasphere Chatbot API Stress Testing"
    echo "============================================="
    echo
    
    # Check API health
    check_api_health
    
    # Determine endpoint type
    local base_endpoint
    case $ENDPOINT in
        "chat"|"chat/stream")
            base_endpoint="chat"
            ;;
        "fast_chat_v2"|"fast_chat_v2/stream")
            base_endpoint="fast_chat_v2"
            ;;
        *)
            print_error "Invalid endpoint: ${ENDPOINT}"
            print_status "Valid endpoints: chat, chat/stream, fast_chat_v2, fast_chat_v2/stream"
            exit 1
            ;;
    esac
    
    # Run tests based on endpoint type
    if [[ "$ENDPOINT" == *"stream"* ]]; then
        # Streaming endpoint
        run_streaming_test "$ENDPOINT" "$REQUESTS"
    else
        # Regular endpoint
        run_ab_test "$ENDPOINT" "$REQUESTS" "$CONCURRENT"
        run_curl_stress_test "$ENDPOINT" "$REQUESTS" "$CONCURRENT"
    fi
    
    echo
    print_success "All tests completed! Check the result files for detailed statistics."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --requests)
            REQUESTS="$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        --url)
            API_BASE_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --endpoint ENDPOINT    Endpoint to test (default: chat)"
            echo "  --requests N           Number of requests (default: 1000)"
            echo "  --concurrent N         Number of concurrent requests (default: 50)"
            echo "  --url URL             API base URL (default: http://localhost:8000)"
            echo "  --help, -h            Show this help message"
            echo
            echo "Examples:"
            echo "  $0 --endpoint chat --requests 500 --concurrent 25"
            echo "  $0 --endpoint fast_chat_v2 --url http://api.example.com"
            echo "  $0 --endpoint chat/stream --requests 200"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_status "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"

