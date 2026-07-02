import os
from catalogue import load_catalogue
from embeddings import build_assessment_embeddings
from models import ChatMessage
from agent import run_agent

catalogue = load_catalogue()
embeddings = build_assessment_embeddings(catalogue)

# Test 1: Vague request
print("=== Test 1: Vague Request ===")
messages_1 = [
    ChatMessage(role="user", content="I want an assessment.")
]
try:
    res1 = run_agent(messages_1, catalogue, embeddings)
    print("Reply:", res1.reply)
    print("Recommendations count:", len(res1.recommendations))
except Exception as e:
    print("Error:", e)

# Test 2: Specific request
print("\n=== Test 2: Specific Request ===")
messages_2 = [
    ChatMessage(role="user", content="I want to test a Java Developer on core skills and coding.")
]
try:
    res2 = run_agent(messages_2, catalogue, embeddings)
    print("Reply:", res2.reply)
    print("Recommendations:")
    for r in res2.recommendations:
        print(f"- {r.name} ({r.test_type}): {r.url}")
except Exception as e:
    print("Error:", e)

# Test 3: Catalog mapping verification (unit tests)
print("\n=== Test 3: Catalog Mapping Verification ===")
from agent import find_catalog_item, get_test_type
# Test exact name match (case-insensitive)
item1 = find_catalog_item("Core Java (Entry Level) (New)", "http://hallucinated-url.com", catalogue)
print("Exact Name Match:", item1["name"] if item1 else "None")

# Test exact URL match
item2 = find_catalog_item("Fake Name", "https://www.shl.com/products/product-catalog/view/java-8-new/", catalogue)
print("Exact URL Match:", item2["name"] if item2 else "None")

# Test substring name match
item3 = find_catalog_item("Java 8", "http://another-fake.com", catalogue)
print("Substring Name Match:", item3["name"] if item3 else "None")

# Test test_type classification
if item2:
    print("Java 8 Test Type:", get_test_type(item2))  # Expected: K
opq_item = next((item for item in catalogue if "opq" in item["name"].lower()), None)
if opq_item:
    print(f"{opq_item['name']} Test Type:", get_test_type(opq_item))  # Expected: P

