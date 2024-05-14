import matplotlib.pyplot as plt

file_path = 'data/out/merged.txt'


word_freq = {}


with open(file_path, 'r') as file:
    for line in file:
        if line.strip():
            word, freq = line.strip().strip("()").split(', ')
            word_freq[word.strip("'")] = int(freq)

data = list(word_freq.items())


data_sorted = sorted(data, key=lambda x: x[1], reverse=True)

N = 50 
top_words = [word[0] for word in data_sorted[:N]]
frequencies = [word[1] for word in data_sorted[:N]]

plt.figure(figsize=(10, 6))
plt.bar(top_words, frequencies, color='skyblue')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.title('Top {} Palabras de Mayor Frecuencia'.format(N))
plt.xticks(rotation=45)
plt.tight_layout()


plt.show()