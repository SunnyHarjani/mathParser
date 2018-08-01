# mathParser
CROHME Submission

Uses a random forest classifier trained on features obtained through segmented linear regression to classify handwritten math symbols

* There are 4 user-modifiable variables defined at the top:
	* trainingSplit: 
		* Decimal value representing how many .inkml files should be partitioned into the 
		training set. I.e. trainingSplit = 0.7 means a 70/30 split for training/testing
	* xSegments, ySegments:
		* Number of segmentations to perform for feature extraction
	* resubstitution:
		* Boolean value that sets testingData equal to trainingData
* Command Line arguments:
	* symbolsDir: The first (optional) command line argument can be a path to a folder containing the .inkml
				it should use for training and testing. This parameter is ignored if cached .pickle files are found.
* Saved parameters:
	* This program saves the two trained classifiers as KDTree.pickle and RandomForest.pickle.
	If these files are present in the local directory, the program will use them. Otherwise,
	it will create new classifiers and save them to disk automatically.
	* This program checks for pre-existing training data and testing data named trainingData.pickle
	and testingData.pickle. If these files are not found, it will partition the 'symbols' 
	dictionary using the ratio defined by 'trainingSplit' and extract the features from them as defined by 
	xSegments, ySegments
		* The symbols dictionary attempts to populate itself from a pre-existing symbols file named
		symbols.pickle. If that file doesn't exist, it will generate a new symbols dictionary
		using a directory provided. The program searches for a file named 'iso_GT.txt' in the 
		symbols directory to optionally label the symbols it reads.
* If this program finds trainingData.pickle and testingData.pickle, it doesn't need any symbols.
* If this program needs symbols and it finds symbols.pickle, it doesn't need any .inkml files.
* If either of these cases are true, symbolsDir does not need to be set.
* If you want to use new .inkml files, delete symbols.pickle, testingData.pickle, and trainingData.pickle.
* If you want to change the training/test split on pre-cached data, delete testingData.pickle and trainingData.pickle.
* If you want to retrain the classifiers, delete KDTree.pickle and RandomForest.pickle.
