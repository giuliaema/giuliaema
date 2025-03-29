#i have chosen the next libraries because at first when i tried with k-values i didn't like that i had to choose how many groups it created, the algorithm should do that automatically
#cause if i were to put it to generate 5 groups, it would force some of the files in the same group (like the score would be as low as 0.21??), the dbscan didn't work entirely but it worked better than k-values
#cause it would sort similar groups, but it couldn't see the similitudes between some files, like it would put a lot of files in their group all alone
#but upon further suspection i would find that there were some clear similarities
#plus when expecting the give files i had found out that some sites were "different"
#in the sense that the text wasn't identical, but it was paraphrased, sth that could
#be detected by a user immediately
from sentence_transformers import SentenceTransformer#to convert the textual content of the HTML files into vector embeddings(to capture 
#the semantic meaning of the text, in case sites talk about similar things but put a little differently, rather than having 
#the same exact strings)
from sklearn.metrics.pairwise import cosine_similarity#to measure how similar vectors(textual content of documents, sentences or words)  are by comparing their direction (not their magnitude)
#highlighting the similar content
from bs4 import BeautifulSoup#to parse through the documents;it takes the raw html content
import os#for reading files and walking through directories.
import json#for saving the results in JSON format to keep the output structured.
from collections import defaultdict#to provides a default value for nonexistent keys(no need to check whether a key exists before modifying it.
# function to read and parse HTML files
def read_html_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:  # opens the HTML file in read mode with UTF-8 encoding
            content = file.read()  # reads the entire content of the HTML file
            soup = BeautifulSoup(content, "html.parser")  # to extract the HTML structure
            # extracts title (if it exists), headings, links and body text from the HTML content
            title = soup.title.string if soup.title else ""  # extracts the title tag if it exists (or we use an empty string)
            headings = " ".join([h.get_text() for h in soup.find_all(["h1", "h2", "h3"])]).strip()  # extracts the text from those tags and then joins it
            links = " ".join([a.get_text() for a in soup.find_all("a")]).strip()  # extracts the visible link text from all <a> tags
            text = soup.get_text().strip()  # extracts all the text from the document (without any HTML tags)
            # for easier processing
            return f"{title}\n{headings}\n{links}\n{text}"
    except Exception as e:
        print(f" Error reading {filepath}: {e}")  # prints any error that might appear
        return ""  # returns an empty string if there's an error (this way we avoid proceeding with corrupted data)
    
# function to load HTML files from a directory
def load_html_files_from_directory(directory):
    html_files = []  #  store valid HTML files
    for root, dirs, files in os.walk(directory):  # walks through the directory recursively to find all files
        for filename in files:
            file_path = os.path.join(root, filename)  # gets the full path to the file
            if filename.endswith('.html'):  # we check if the file has an HTML extension
                text = read_html_file(file_path)  # for reading the content of the HTML file
                if text.strip():  # we ensure that the file contains non-empty text
                    html_files.append((filename, text))  
    return html_files  # it returns the list of HTML files with their parsed content

# function to generate BERT embeddings for documents
def generate_bert_embeddings(documents):
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # i have chosen this for its efficiency and speed, plus it's trained for semantic textual similiarity and it's not as computationally expensive as bert-base-uncased
        embeddings = model.encode(documents)  # converts the list of document texts into embeddings (vectors of numbers)
        return embeddings  # returns the generated embeddings for comparison
    except Exception as e:
        print(f" Error loading model: {e}")  # handles errors during model loading (e.g., network or missing model issues)
        return None  # returns None if the model fails to load

# function to compare HTML files based on their embeddings
def compare_html_files(html_files, similarity_threshold=0.70):
    documents = [text for _, text in html_files]  # extracts the text content from each file 
    embeddings = generate_bert_embeddings(documents)  # generates embeddings for the documents using the BERT model (it is good at detecting paraphrasing)
    if embeddings is None or len(embeddings) < 2:
        print("Not enough documents for meaningful comparison.")  # to handle insufficient data for comparison
        return []
    similarity_matrix = cosine_similarity(embeddings)  # it calculates a matrix containing the cosine similarity between each pair of document embeddings ()
    similar_files = defaultdict(list)  # here we use defaultdict to store lists of similar files for each file (to avoid the need for explicit checks)

    # to iterate over all pairs of documents ( to check for their similarity)
    for i in range(len(html_files)):
        for j in range(i + 1, len(html_files)):  # only compares each pair once by iterating over the upper triangle of the similarity matrix
            similarity_score = float(similarity_matrix[i][j])  # to extract the similarity score between document i and document j
            if similarity_score > similarity_threshold:  # onluyconsiders files with a similarity score greater than the threshold (which i have chosen to be 0.70)
                file1, file2 = html_files[i][0], html_files[j][0]  # gets the filenames of the two compared documents
                # adds the similar files er in the defaultdict, storing them both in each other's lists
                similar_files[file1].append((file2, similarity_score))
                similar_files[file2].append((file1, similarity_score))
    
    # converts defaultdict to a list of groups (that way it ensures no files are visited twice)
    grouped_files = []  #to store groups of similar files
    visited = set()  #  to track the already processed files 
    
    # for grouping the similar files together
    for file, matches in similar_files.items():
        if file not in visited:  # it skips files that have already been processed
            group = [file]  # it starts a new group with the current file
            visited.add(file)  # it marks this file as processed
            # we add all other files that are similar to this one to the group
            for match, score in matches:
                if match not in visited:
                    group.append(match)
                    visited.add(match)
            grouped_files.append(group)  # it adds the newly formed group of similar files
    return grouped_files, similarity_matrix  # it returns both the grouped files and the full similarity matrix for reference

# function to save and print the results
def save_and_print_results(grouped_files, output_file="similar_files.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        # it saves the grouped files in JSON format (to make the output easier to read and analyze later)
        json.dump(grouped_files, f, indent=4)
    print("\n Grouped Similar Documents:")  # to print the results to the console
    for group in grouped_files:
        print(f" Group: {', '.join(group)} \n")  # it displays each group containing the similar documents
        
# main function to execute the full process
def main(directory, similarity_threshold=0.75):
    html_files = load_html_files_from_directory(directory)  # to load all HTML files from the specified directory
    # to handle cases where no valid HTML files are found
    if not html_files:
        print("No valid HTML files found in the directory!!")  # for the case in which no HTML files are found
        return
    # here we compare the HTML files based on their semantic content
    grouped_files, similarity_matrix = compare_html_files(html_files, similarity_threshold)
    # if no significant similarities are found, it notifies the user
    if not grouped_files:
        print("No significant similarities detected.")
    else:
        save_and_print_results(grouped_files)  # saves and prints the results of the comparison
        
if __name__ == "__main__":
    directory = "/Users/giuliaemanoil/Downloads/clones" 
    similarity_threshold = 0.70 #i put this threshold thinking that it is not too low for the sites too be considered similar (0.75 was too high and 0.65 was too low imo)
    main(directory, similarity_threshold)  