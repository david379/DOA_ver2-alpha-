# DOA_ver2-alpha-
Implemented multiprocessing

This version of the code contains multiprocessing in the "test.py" file. It obtains real time results, but you have to be careful while running it. Because of the multiprocessing, the calculation now has to happen in a "main" file and not inside a function, so to obtain the DOA, run the "test" file once, then put all of the code inside the "if __name__ == '__main__'" in a comment, and uncomment the second half of the code in the file in order to finish the processing of the signal. 
