"""Generates TOC from example.html"""

import toc
import re

#HEADING_LEVELS = '3-5'
#toc.HEADING_RE = re.compile('<h(?P<level>[%(level)s])\\s?(?:id=["\']'\
#    '(?P<id>.*?)["\'])?\\s?>(?P<text>.*?)</h[%(level)s]>' \
#        % { "level": HEADING_LEVELS })

def generate_example():

    #Do the latex conversion
    import eqhtml
    eqhtml.embed_latex_in_html("../docs/user_guide_source.html", "../docs/user_guide.html")

    #---- Put today's date ----
    from datetime import date
    d = date.today()
    f = open("../docs/user_guide.html")
    html = f.read()
    html = html.replace("{{date}}", d.strftime("%B %d, %Y"))
    f.close()

    #---- Make table of contents ----
    import toc
    toc_inst = toc.Toc()
    print "converting html of length", len(html)
    file_output = open("../docs/user_guide.html", 'w')
    file_output.write(
        toc_inst.toc_template( '{{toc}}', html, prefix_li=False )
        )
    file_output.close()


if __name__ == '__main__':
    generate_example()

    