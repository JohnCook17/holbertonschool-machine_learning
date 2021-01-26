# **0x13 QA bot**

The objective of this project was to build a fully functional QA bot. I was required to use bert-uncased-tf2-qa model, and the pre-trained BertTokenizer, bert-large-uncased-whole-word-masking-finetuned-squad, from the transformers library. There are 5 objectives in total. 


## **Task**

## 0. Question Answering

		The goal of task 0 is to answer a question for a single text file. This is done by
		first having a question fed to the function along with the path to a text to search.
		If no answer if found it should return None, otherwise it will return an answer.
		I used the following guide to start off with:
		https://tfhub.dev/see--/bert-uncased-tf2-qa/1
		
## 1. Create the loop
		This is a simple input output loop, I think the code speaks for itself here. The loop 
		will exit if you type the one of the following words "exit, quit, goodbye, bye".

## 2. Answer Questions
		This task combines the two previous task into a simple single document qa bot.
		It will reply "Sorry, I do not understand your question." if a question is not
		answered in the document.

## 3. Semantic Search
		This task performs a simular process to task 0 but it is across multiple files.
		corpus_path is the path to multiple text that need to be searched, and sentence
		is the sentence to search for. I did some fine tuneing on the process, namely I
		only care about how similar a text is not how dissimilar it is. So I turn all the
		negative numbers into 0's. I then drop the 0's and change the minimum value to 6.755.
		The reason for this is that 6.755 is about the average match of any question asked
		for any particular document. I then divide by the number of entries if there is more
		then one. This process allows the weight of sentences pertaining to the given
		sentence to have more weight and not drag down the average matching score. While
		still counting their weight.
		
## 4. Multi-reference Question Answering
		This task is a culmination of all previous task. First I enter the question and the
		loop begins. It then performs Semantic Search on multiple files. I wish this process
		could be faster but it takes some time to search through all 91 files. Once a file is
		determined to be likely to contain the answer it is fed to text search, which
		searches the file for the answer. The answer is then displayed on screen and the loop
		continues untill it is closed.