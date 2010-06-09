# -*- coding: utf-8 -*-

"""
toc.py by Stuart Colville http://muffinresearch.co.uk
License: http://www.opensource.org/licenses/mit-license.php
Created: 2008-08-28

"""

import re

import smallwords
import odict

BeautifulSoup = None

LANG = "en"

WORD_RE = re.compile(r"(?<!\s[\d],)[\s”“\".,:;{}()\[\]\?]+")
HEADING_RE = re.compile('<h(?P<level>[3-5])\\s?(?:id=["\']'\
    '(?P<id>.*?)["\'])?\\s?>(?P<text>.*?)</h[3-5]>')
    
LI_STRING = '<li>%(prefix)s<a href="#%(key)s">%(text)s</a>'
ID_PREFIX = ""
ID_SUFFIX = ""
OL_CLASSNAME = 'toc'
DEBUG = False

class Toc:
    """Toc generates a table of contents in the form of an ordered list 
    
    containing fragment identifiers of all of the h1-h5 in the html document.

    If the heading has a id already this is used to generate the link if it
    doesn't then a unique identifier is generated based on the heading text with
    an optional prefix and or suffix.

    Regex for replacement sets sensible defaults but if you want to do 
    something different simply override with your own regex object. See 
    generate_example.py for an example of this.
    
    """
    
    def __init__(self):
        self.toc_data = odict.OrderedDict()
        self.last_prefix = ''
        self.heading_count = {}
        self.previous_level = self.current_level = 0
        
    @staticmethod
    def slugify(text):
        """Splits words and then joins the lower cased words with hyphens"""

        words = re.split(WORD_RE, text)
        title = []
        for word in words:
            if word.lower() not in getattr(smallwords, LANG, smallwords.en):
                title.append(word.lower())
        return "-".join(title)        
    
    def list_item_prefix(self):
        
        """Generates a prefix for lists
        
        The format is:
        
        1 A very nice title
            1.1 Another heading
            1.2 Awesomeness
        2 This is a fantastic heading
            2.1 Heading of WIN
                2.1.1 Nested further down
                2.1.2 Some thing else
            2.2 Yet more cool stuff
            2.3 And some more
            
        """
        
        prefix_list = []
        if len(str(self.last_prefix)) > 0:
            prefix_list = str(self.last_prefix).split('.')
#            if len(prefix_list[-1])==0:
#                del prefix_list[-1]
            
        if not self.last_prefix:
            prefix = "1"
            prefix_list = ['1']
        
        # current level is higher than the last e.g h2->h3 1.1->1.1.1
        elif (self.current_level > self.previous_level):
            prefix = "%s.1" % self.last_prefix
        
        # current level is the same as the last h3=h3 1.1.1->1.1.2
        elif self.current_level == self.previous_level:
            if len(prefix_list)==1:
                prefix = "%s" % (int(prefix_list[0]) + 1)
            else:
                prefix = "%s.%s" % (
                    ".".join(prefix_list[:-1]),
                    int(prefix_list[-1]) + 1
                )

        # current level is lower thant the last  h3->h2 1.2->2
        elif self.current_level < self.previous_level:
            difference = self.previous_level - self.current_level 
            reduced = prefix_list[:-difference]

            if len(reduced) ==  1:
                prefix = int(prefix_list[0]) + 1
            elif len(reduced) >  1:                
                prefix = "%s.%s" % (
                    ".".join(prefix_list[:-(difference+1)]), 
                    int(prefix_list[:-difference][-1]) + 1
                )
            
        self.last_prefix = str(prefix)
        
        if DEBUG:
            print "PREFIX: %s" % prefix
            print "-------------------------"
        
        return str(prefix) + "."
    
    
    def heading_repl(self, match):
        
        """Replaces the headings with generated ids
        
        this is the callback to the re.sub in generate_toc. The nice thing 
        about how this works is by saving our data we are essentially doing a 
        one pass find and replace *and* at the same time we are stashing 
        the necessary data in class vars
        
        """
        
        # Get the necessary parts from the match "_obj
        level = match.group("level")
        identifier = match.group("id") or None
        text = match.group("text") or None

        self.current_level = int(level)
        prefix = self.list_item_prefix()

        if identifier is None:
            
            text = "%s%s%s" % (ID_PREFIX, text, ID_SUFFIX,)
            original_text = text
            
            # Test if the first char is a number if so prefix with "_"
            if re.match('\d', text[1]):
                text = "_%s" % text
            
            identifier = self.slugify(text)
        
        # Deal with duplicate ids by appending _n to the end of the id
        # whilst duplicates are found. Unlikely to be a situation where
        # this while loop continues for too long.
        version = 1
        while self.toc_data.get(identifier, None) is not None:
            version = version + 1
            identifier = "%s_%s" % (re.sub('_\d+$', '', identifier), version,)
        
        self.toc_data[identifier] = {
            "level": int(level),
            "text": original_text,
        }

        self.previous_level = self.current_level

        return "<h%s id='%s'>%s %s</h%s>" % (
            level,
            identifier,
            prefix,
            original_text,
            level,
        )
    
    def generate_toc(self, html, prefix_li=False):
        
        """Generates and table of contents from html
        
        Returns a tuple containing modified html and an ordered list of
        contents based on the headings.
        
        """
        
        # replaces created via callback self.heading_repl
        html = re.sub(HEADING_RE, self.heading_repl, html)
        
        toc_ol = ['<ol>\n']
        
        self.previous_level = -1000
        
        for key in self.toc_data:
            
            self.current_level = self.toc_data[key]['level']
            text = self.toc_data[key]['text']
                        
            if (self.current_level > self.previous_level) and len(toc_ol) > 1:
                toc_ol.append('\n<ol>\n')
            if self.current_level == self.previous_level:
                toc_ol.append('</li>\n')
            if self.current_level < self.previous_level:
                difference = self.previous_level - self.current_level 
                toc_ol.append(difference * '</li>\n</ol>\n') #</li>\n')

            prefix = ''
            toc_ol.append(LI_STRING % {
                "prefix" : prefix,
                "key" : key, 
                "text": text,
            })
            self.previous_level = self.current_level
        
        toc_ol.append(self.current_level * '</li>\n</ol>\n')
        self.previous_level = self.current_level = 0
        
        return (html, ''.join(toc_ol))
        
    def toc_template(self, template_var, html, **kwargs):
        """Provides a way to template the insertion of the toc.
        
        Second argument is a regex to match outputs the html with the toc 
        replacing the string defined in template_var.
        
        For example if my html looks like:
        
        <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
            "http://www.w3.org/TR/html4/strict.dtd">
        <html>    
            <head>
                <style type="text/css" media="screen">

                </style>
            </head>
            <body>
                <h1>Lorem ipsum dolor sit amet<h1>
                {{toc}}
                <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit, 
                sed do eiusmod tempor incididunt ut labore et dolore magna</p> 

                <h2>Consectetur adipisicing elit</h2>
                <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit, 
                sed do eiusmod tempor incididunt ut labore et dolore magna</p>
                
                <h3>Sed do eiusmod tempor</h3>
                <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit, 
                sed do eiusmod tempor incididunt ut labore et dolore magna</p>
                
                <h2>Ut enim ad minim veniam</h2>
                <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit, 
                sed do eiusmod tempor incididunt ut labore et dolore magna</p>
            </body>
        </html>
        
        I'll run:
        
        t = toc.Toc()
        t.toc_template('{{toc}}', html)
        
        """
        prefix_li = kwargs.get('prefix_li', False)
        toc = self.generate_toc(html, prefix_li)
        toc_output = BeautifulSoup is not None and\
            BeautifulSoup(toc[1]).prettify() or toc[1]
        
        return  toc[0].replace(template_var, toc_output)





if __name__ == '__main__':
    toc_inst = Toc()
    file_output = open('../docs/user_guide.html', 'w')
    file_output.write(
        toc_inst.toc_template(
            '{{toc}}',
            open('../docs/user_guide_source.html').read(),
            prefix_li=True
        )
    )
    file_output.close()
