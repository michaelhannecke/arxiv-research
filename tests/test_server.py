#!/usr/bin/env python3
"""Test script to verify the FastAPI server is working"""

import httpx
import json

def test_server():
    base_url = "http://localhost:8000"
    
    print("Testing arXiv Document Processor API...")
    print("-" * 50)
    
    # Test health endpoint
    print("\n1. Testing /health endpoint...")
    response = httpx.get(f"{base_url}/health")
    if response.status_code == 200:
        print("   ✓ Health check passed")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")
    
    # Test root endpoint
    print("\n2. Testing / endpoint...")
    response = httpx.get(base_url)
    if response.status_code == 200:
        print("   ✓ Root endpoint accessible")
        print(f"   Response contains HTML: {len(response.text)} characters")
    else:
        print(f"   ✗ Root endpoint failed: {response.status_code}")
    
    # Test API docs
    print("\n3. Testing /docs endpoint...")
    response = httpx.get(f"{base_url}/docs")
    if response.status_code == 200:
        print("   ✓ API docs accessible")
    else:
        print(f"   ✗ API docs failed: {response.status_code}")
    
    # Test process endpoint with sample arXiv ID
    print("\n4. Testing /api/process endpoint...")
    test_arxiv_id = "2301.00001"
    response = httpx.post(
        f"{base_url}/api/process",
        json={"arxiv_id": test_arxiv_id}
    )
    if response.status_code == 200:
        result = response.json()
        print("   ✓ Process endpoint working")
        print(f"   Task ID: {result['task_id']}")
        print(f"   Status: {result['status']}")
        
        # Test status endpoint
        print("\n5. Testing /api/status endpoint...")
        status_response = httpx.get(f"{base_url}/api/status/{result['task_id']}")
        if status_response.status_code == 200:
            print("   ✓ Status endpoint working")
            print(f"   Task status: {status_response.json()['status']}")
        else:
            print(f"   ✗ Status endpoint failed: {status_response.status_code}")
    else:
        print(f"   ✗ Process endpoint failed: {response.status_code}")
    
    # Test history endpoint
    print("\n6. Testing /api/history endpoint...")
    response = httpx.get(f"{base_url}/api/history")
    if response.status_code == 200:
        print("   ✓ History endpoint working")
        print(f"   History items: {len(response.json())}")
    else:
        print(f"   ✗ History endpoint failed: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    try:
        test_server()
    except httpx.ConnectError:
        print("ERROR: Could not connect to server.")
        print("Make sure the server is running with:")
        print("  uvicorn app.main:app --reload --port 8000")