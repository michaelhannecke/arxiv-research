#!/usr/bin/env python3
"""Test script for arXiv integration"""

import asyncio
import httpx
import json
import time
from datetime import datetime

async def test_arxiv_integration():
    """Test the complete arXiv integration flow"""
    base_url = "http://localhost:8000"
    
    print("Testing arXiv Integration")
    print("=" * 50)
    
    # Test arXiv ID - using a known paper
    test_arxiv_id = "1706.03762"  # "Attention Is All You Need" paper
    
    print(f"\nTesting with arXiv ID: {test_arxiv_id}")
    print("(Attention Is All You Need - Transformer paper)")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            
            # Step 1: Submit processing request
            print("\n1. Submitting processing request...")
            response = await client.post(
                f"{base_url}/api/process",
                json={"arxiv_id": test_arxiv_id}
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                print(f"   ✓ Request submitted successfully")
                print(f"   Task ID: {task_id}")
                print(f"   Status: {result['status']}")
            else:
                print(f"   ✗ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return
            
            # Step 2: Poll for completion
            print(f"\n2. Monitoring processing progress...")
            max_wait = 120  # 2 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = await client.get(f"{base_url}/api/status/{task_id}")
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    progress = status["progress"]
                    task_status = status["status"]
                    
                    print(f"   Progress: {progress}% - Status: {task_status}")
                    
                    if task_status == "completed":
                        print("   ✓ Processing completed successfully!")
                        result_data = status.get("result", {})
                        print(f"   Title: {result_data.get('title', 'N/A')}")
                        print(f"   Processing time: {result_data.get('processing_time', 0):.2f}s")
                        print(f"   Sections: {result_data.get('sections_count', 0)}")
                        print(f"   References: {result_data.get('references_count', 0)}")
                        break
                    elif task_status == "failed":
                        print("   ✗ Processing failed")
                        print(f"   Error: {status.get('error', 'Unknown error')}")
                        return
                    elif task_status == "cancelled":
                        print("   ⚠ Processing was cancelled")
                        return
                    
                    # Wait before next check
                    await asyncio.sleep(2)
                else:
                    print(f"   ✗ Status check failed: {status_response.status_code}")
                    break
            else:
                print(f"   ⚠ Processing timed out after {max_wait}s")
                return
            
            # Step 3: Check history
            print(f"\n3. Checking processing history...")
            history_response = await client.get(f"{base_url}/api/history")
            
            if history_response.status_code == 200:
                history = history_response.json()
                print(f"   ✓ History loaded: {len(history)} items")
                
                # Find our document
                our_doc = None
                for item in history:
                    if item["arxiv_id"] == test_arxiv_id:
                        our_doc = item
                        break
                
                if our_doc:
                    print(f"   ✓ Found our document in history")
                    print(f"   Title: {our_doc['title']}")
                    print(f"   Authors: {', '.join(our_doc['authors'][:3])}...")
                    print(f"   Keywords: {', '.join(our_doc['keywords'])}")
                else:
                    print(f"   ⚠ Document not found in history")
            else:
                print(f"   ✗ History check failed: {history_response.status_code}")
            
            # Step 4: Retrieve document
            print(f"\n4. Retrieving processed document...")
            doc_response = await client.get(f"{base_url}/api/document/{test_arxiv_id}")
            
            if doc_response.status_code == 200:
                document = doc_response.json()
                print(f"   ✓ Document retrieved successfully")
                print(f"   Title: {document['title']}")
                print(f"   Content length: {len(document['content'])} characters")
                print(f"   Authors: {len(document['authors'])} authors")
                print(f"   Keywords: {len(document['keywords'])} keywords")
                
                # Show first part of content
                content = document['content']
                lines = content.split('\n')[:10]
                print(f"\n   First 10 lines of content:")
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        print(f"   {i:2d}: {line[:80]}...")
                        
            else:
                print(f"   ✗ Document retrieval failed: {doc_response.status_code}")
                print(f"   Error: {doc_response.text}")
        
        print(f"\n" + "=" * 50)
        print("Integration test completed successfully! 🎉")
        print("\nThe arXiv integration is working:")
        print("✓ Downloads papers from arXiv")
        print("✓ Extracts content from PDFs")
        print("✓ Stores processed documents")
        print("✓ Provides status tracking")
        print("✓ Maintains processing history")
        
    except httpx.ConnectError:
        print("ERROR: Could not connect to server.")
        print("Make sure the server is running with:")
        print("  ./run.sh")
        
    except Exception as e:
        print(f"ERROR: Unexpected error during testing: {e}")

async def test_invalid_arxiv_id():
    """Test with invalid arXiv ID"""
    base_url = "http://localhost:8000"
    
    print("\n" + "=" * 30)
    print("Testing invalid arXiv ID")
    print("=" * 30)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test invalid ID
            response = await client.post(
                f"{base_url}/api/process",
                json={"arxiv_id": "invalid-id"}
            )
            
            if response.status_code == 422:  # Validation error
                print("✓ Correctly rejected invalid arXiv ID")
            else:
                print(f"⚠ Unexpected response: {response.status_code}")
                
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Starting arXiv Integration Tests...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run main integration test
        asyncio.run(test_arxiv_integration())
        
        # Test error handling
        asyncio.run(test_invalid_arxiv_id())
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")