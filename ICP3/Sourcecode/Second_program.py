#importing BeautifulSoup library
from bs4 import BeautifulSoup

#importing urlopen library
from urllib.request import urlopen
url="https://en.wikipedia.org/wiki/Deep_learning"
html= urlopen(url)

#html.parser access only tags
data=BeautifulSoup(html, 'html.parser')

#printing title of the page
print(data.title.string)

#finding all the a tagged
web_links=data.find_all('a')

# Opening a file in append mode
text_file=open('Weblinks.txt','a')

# writing all the links in the Weblinks.txt
for link in web_links:
    text_file.write(str(link.get('href')))
    text_file.write('\n')
text_file.close()



