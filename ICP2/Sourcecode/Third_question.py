# PART ONE

L = ["Python Course\n", "Deep Learning Course\n"] #input file

# writing to file
file1 = open('kalyani_file.txt', 'w') # save an empty .txt file in your working directory
file1.writelines(L) # then write to that empty file
file1.close()

#PART TWO
my_dict = {} # defined an empty dictionary
# opening the text file
with open('kalyani_file.txt','r') as file:

    # reading each line
    for line in file:

        # reading each word
        for word in line.split():

            #remember that each unique word becomes a dictionary key

            if word in my_dict.keys():
                my_dict[word] = my_dict[word]+1 # counter for each word
            else:
                my_dict[word]=1

#print output
for elem in my_dict:
    print('{} : {}'. format(elem, my_dict[elem]))