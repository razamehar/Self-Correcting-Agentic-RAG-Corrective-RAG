from pprint import pprint
from agentic_rag_workflow import app


query = input("What is your question? ")
question = {"question": query}

final_response = None

for output in app.stream(question):
    for key, value in output.items():
        print(f"Node '{key}':")
        if "generation" in value:
            final_response = value["generation"]
            pprint(final_response)

if final_response is not None:
    print("\nGenerating the final response:")
    print(final_response)
else:
    print("No generation found in any node output.")
