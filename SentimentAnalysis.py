from transformers import pipeline

# Define a function to analyze emotions for a given text
def analyze_text_emotions(text):
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    results = pipe(text, top_k=None)
    top_3_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    return [(res['label'], res['score']) for res in top_3_results]

if __name__ == "__main__":
    while True:
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            break

        results = analyze_text_emotions(user_input)
        for label, score in results:
            print(f"Label: {label}, Score: {score:.4f}")
        print()
