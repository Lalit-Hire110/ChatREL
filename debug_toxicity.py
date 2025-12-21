from chatrel.message_processor import extract_toxicity_score

pred = {"label": "non-toxic", "score": 0.97}
result = extract_toxicity_score(pred)
print(f"Input: {pred}")
print(f"Result: {result}")
print(f"Expected: 0.03")

# Debug the function
label = pred.get("label", "").lower()
score = pred.get("score", 0.5)
print(f"\nDebug:")
print(f"  label (lowered): '{label}'")
print(f"  score: {score}")

toxic_labels = {"toxic", "hate", "abusive", "offensive"}
print(f"  Toxic check: {any(w in label for w in toxic_labels)}")
print(f"  'non' in label: {'non' in label}")
print(f"  label == 'clean': {label == 'clean'}")

if any(w in label for w in toxic_labels):
    print(f"  -> Would return score: {score}")
elif "non" in label or label == "clean":
    print(f"  -> Would return 1.0 - score: {1.0 - score}")
else:
    print(f"  -> Would use fallback: {score}")
