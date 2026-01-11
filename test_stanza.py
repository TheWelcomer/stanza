import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,morphseg', download_method=None)
doc = nlp("According to all known laws of aviation, there is no way a bee should be able to fly.")

# Print morpheme segmentations
print("\nMorpheme Segmentations:")
print("=" * 60)
for sentence in doc.sentences:
    for word in sentence.words:
        if hasattr(word, 'morphemes'):
            morphemes = ' + '.join(word.morphemes)
            print(f"{word.text:15} -> {morphemes}")
        else:
            print(f"{word.text:15} -> NO MORPHEMES ATTRIBUTE")