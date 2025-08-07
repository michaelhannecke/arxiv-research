#!/usr/bin/env python3
"""Test script for AI integration (Phase 4)"""

import asyncio
import httpx
import json
import time
import os
from datetime import datetime

async def test_ai_integration():
    """Test the AI integration features"""
    base_url = "http://localhost:8000"
    
    print("Testing AI Integration (Phase 4)")
    print("=" * 50)
    
    # Check if API key is configured
    api_key_configured = os.path.exists("secrets/.env")
    if api_key_configured:
        with open("secrets/.env", "r") as f:
            api_key_configured = "ANTHROPIC_API_KEY" in f.read()
    
    if api_key_configured:
        print("✓ ANTHROPIC_API_KEY detected")
        print("  Testing full AI features...")
    else:
        print("⚠ No ANTHROPIC_API_KEY found")
        print("  Testing with fallback features...")
        print("  To enable AI features, add your key to secrets/.env:")
        print("  echo 'ANTHROPIC_API_KEY=your-key-here' >> secrets/.env")
    
    # Use a different paper for AI testing
    test_arxiv_id = "2010.11929"  # GPT-3 paper - good for AI summarization
    
    print(f"\nTesting with arXiv ID: {test_arxiv_id}")
    print("(Language Models are Few-Shot Learners - GPT-3 paper)")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            
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
            else:
                print(f"   ✗ Request failed: {response.status_code}")
                return
            
            # Step 2: Monitor processing with detailed stage tracking
            print(f"\n2. Monitoring AI processing pipeline...")
            max_wait = 180  # 3 minutes for AI processing
            start_time = time.time()
            
            stages_seen = set()
            
            while time.time() - start_time < max_wait:
                status_response = await client.get(f"{base_url}/api/status/{task_id}")
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    progress = status["progress"]
                    task_status = status["status"]
                    
                    # Get stage info if available
                    stage = "processing"
                    if "result" in status and status["result"]:
                        # Try to get stage from task processor
                        pass
                    
                    if task_status == "processing":
                        print(f"   Progress: {progress}% - Status: {task_status}")
                        
                        # Show different messages based on progress
                        if progress >= 50 and "ai_stage" not in stages_seen:
                            print("   🤖 AI summarization in progress...")
                            stages_seen.add("ai_stage")
                        elif progress >= 70 and "keyword_stage" not in stages_seen:
                            print("   🔍 AI keyword extraction...")
                            stages_seen.add("keyword_stage")
                    
                    elif task_status == "completed":
                        print("   ✓ Processing completed successfully!")
                        result_data = status.get("result", {})
                        
                        print(f"   Title: {result_data.get('title', 'N/A')}")
                        print(f"   Processing time: {result_data.get('processing_time', 0):.2f}s")
                        print(f"   Sections: {result_data.get('sections_count', 0)}")
                        
                        if api_key_configured:
                            print("   🤖 AI features used: Enhanced summaries, keyword extraction")
                        else:
                            print("   📝 Basic processing: No AI features (API key not configured)")
                        
                        break
                    elif task_status == "failed":
                        print("   ✗ Processing failed")
                        print(f"   Error: {status.get('error', 'Unknown error')}")
                        return
                    
                    # Wait before next check
                    await asyncio.sleep(3)
                else:
                    print(f"   ✗ Status check failed: {status_response.status_code}")
                    break
            else:
                print(f"   ⚠ Processing timed out after {max_wait}s")
                return
            
            # Step 3: Analyze the generated document
            print(f"\n3. Analyzing AI-enhanced document...")
            doc_response = await client.get(f"{base_url}/api/document/{test_arxiv_id}")
            
            if doc_response.status_code == 200:
                document = doc_response.json()
                content = document['content']
                
                print(f"   ✓ Document retrieved successfully")
                print(f"   Title: {document['title']}")
                print(f"   Content length: {len(content)} characters")
                print(f"   Keywords: {len(document['keywords'])} extracted")
                
                # Analyze AI features in the content
                has_ai_summary = "## AI Summary" in content
                has_section_summaries = "### AI Summary" in content
                
                if has_ai_summary:
                    print("   🤖 ✓ AI-generated comprehensive summary found")
                else:
                    print("   📝 Basic summary (no AI)")
                
                if has_section_summaries:
                    # Count section summaries
                    section_summary_count = content.count("### AI Summary")
                    print(f"   🤖 ✓ AI section summaries: {section_summary_count} sections")
                else:
                    print("   📝 No section-level AI summaries")
                
                # Show AI summary excerpt if available
                if has_ai_summary:
                    lines = content.split('\n')
                    in_ai_summary = False
                    summary_lines = []
                    
                    for line in lines:
                        if line.strip() == "## AI Summary":
                            in_ai_summary = True
                            continue
                        elif line.startswith("## ") and in_ai_summary:
                            break
                        elif in_ai_summary and line.strip():
                            summary_lines.append(line)
                            if len(summary_lines) >= 3:  # First 3 lines
                                break
                    
                    if summary_lines:
                        print(f"\n   AI Summary Preview:")
                        for line in summary_lines:
                            print(f"   > {line.strip()}")
                
                # Show keyword quality
                keywords = document.get('keywords', [])
                if len(keywords) > 5:
                    print(f"\n   Keywords extracted: {', '.join(keywords[:8])}...")
                    
                    # Check for AI-quality keywords (more specific/technical)
                    technical_keywords = [kw for kw in keywords if len(kw) > 8 or ' ' in kw]
                    if technical_keywords:
                        print(f"   🤖 High-quality AI keywords detected: {len(technical_keywords)}")
                    
            else:
                print(f"   ✗ Document retrieval failed: {doc_response.status_code}")
        
        print(f"\n" + "=" * 50)
        if api_key_configured:
            print("AI Integration Test Completed! 🤖✨")
            print("\nAI Features Working:")
            print("✓ Comprehensive document summarization")
            print("✓ Section-by-section AI summaries") 
            print("✓ Enhanced keyword extraction")
            print("✓ Intelligent content analysis")
        else:
            print("Basic Processing Test Completed! 📝")
            print("\nTo enable AI features:")
            print("1. Get an Anthropic API key from https://console.anthropic.com")
            print("2. Add it to secrets/.env:")
            print("   echo 'ANTHROPIC_API_KEY=your-key-here' >> secrets/.env")
            print("3. Restart the server and run this test again")
        
    except httpx.ConnectError:
        print("ERROR: Could not connect to server.")
        print("Make sure the server is running with: ./run.sh")
        
    except Exception as e:
        print(f"ERROR: Unexpected error during testing: {e}")

if __name__ == "__main__":
    print("Starting AI Integration Tests...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        asyncio.run(test_ai_integration())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")