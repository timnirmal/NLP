"""
Regular Expressions:

Regular expressions are a powerful tool for finding patterns in text.
What is a regular expression?

    A regular expression is a sequence of characters that define a search pattern.
    It is used to find a match between a pattern and a string.
    Regular expressions are used in many different kinds of text processing.
    For example, they are used to find patterns in email addresses,
    phone numbers, and URLs.

    Regular expressions are used in the Python re module.
    The re module provides regular expression functions for searching and replacing.
    The re module also provides the re.sub() function for performing substitutions.

    Stuff from Copilot:
        Regular expressions are used in the Python string module.
        The string module also provides the string.replace() function for performing substitutions.
        Regular expressions are used in the Python sre_compile module.
        The sre_compile module also provides the sre_compile.sub() function for performing substitutions.


Identifiers:

\d = any number
\D = anything but a number
\s = space
\S = anything but a space
\w = any letter
\W = anything but a letter
. = any character, except for a new line
\b = space around whole words
\. = period. must use backslash, because . normally means any character.

Modifiers:
    {1,3} = for digit, you expect 1-3 count of digital numbers or "places"
    + = match 1 or more
    ? = match 0 or 1 repetitions
    * = match 0 or more repetitions
    $ = match the end of the string
    ^ = match the beginning of the string
    | = match either of the patterns
    [] = range or variance, [a-z] = match characters between a to z (both included)
    {x} = expect to see this amount of the preceding code  / exact number of repetitions
    {x,y} = expect to see this x to y amount of the preceding code

White Space Charts:

\n = new line
\s = space
\t = tab
\e = escape
\f = form feed
\r = carriage return
Characters to REMEMBER TO ESCAPE IF USED!

. + * ? [ ] $ ^ ( ) { } | \
Brackets:

[] = quant[ia]tative = will find either quantitative, or quantatative.
[a-z] = return any lowercase letter a-z
[1-5a-qA-Z] = return all numbers 1-5, lowercase letters a-q and uppercase A-Z

"""

import re

exampleString = '''
Jessica is 15 years old, and Daniel is 27 years old.
Edward is 97 years old, and his grandfather, Oscar, is 102. 
'''


# Extra 1 :  RegEx for finding the name of the person


def find_name(text):
    name = re.findall(r"\b[A-Z]\w+\b", text)
    # name = re.findall(r'[A-Z][a-z]*',exampleString)
    return name

def person_name_finder(text):
    name = find_name(text)
    age = find_age(text)
    person_name = []
    for i in range(len(name)):
        person_name.append(name[i] + " is " + age[i] + " years old.")
    return person_name

def find_age(text):
    age = re.findall(r"\d{1,3}", text)
    return age


print(find_name(exampleString))
print(find_age(exampleString))
print(person_name_finder(exampleString))



# Parse a Website with regex and urllib Python

import urllib.request

#url = 'http://pythonprogramming.net/parse-website-using-regular-expressions-urllib/'
url = 'https://en.wikipedia.org/wiki/Scientist'

request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
resData = response.read()

print(resData)

paragraphs = re.findall(r'<p>(.*?)</p>', str(resData))

for eachP in paragraphs:
    print(eachP)

print(find_name(str(paragraphs)))
