"""A simple tool for embedding LaTeX in XHTML documents.

This script lets you embed LaTeX code between <div> and <span> tags. Example:
    <div class="eq>
      y = \int_0^\infty \gamma^2 \cos(x) dx 
    </div>
    <p> An inline equation <span class="eq">y^2=x^2+\alpha^2</span> here.</p>

The script extracts the equations, creates a temporary LaTeX document, 
compiles it, saves the equations as images and replaces the original markup 
with images.

Usage:
    python eqhtml.py source dest
    
Process source and save result in dest. Note that no error checking is 
performed. 
"""

import xml.etree.ElementTree as et
import os, sys


def embed_latex_in_html(sourcefn, destfn):

    # Include your favourite LaTeX packages and commands here
    tex_preamble = r'''
    \documentclass{article}
    \usepackage{amsmath}
    \usepackage{amsthm}
    \usepackage{amssymb}
    \usepackage{bm}
    \newcommand{\mx}[1]{\mathbf{\bm{#1}}} % Matrix command
    \newcommand{\vc}[1]{\mathbf{\bm{#1}}} % Vector command
    \newcommand{\T}{\text{T}}                % Transpose
    \pagestyle{empty}
    \begin{document}
    '''

    imgpath = 'eq/' # path to generated equations. e.q 'img/'

    # get source and dest filenames from command line

    sourcefn_base = os.path.splitext(os.path.basename(sourcefn))[0]
    # change working directory to the same as source's
    cwd = os.getcwd()
    os.chdir(os.path.abspath(os.path.dirname(sourcefn)))
    sourcefn = os.path.basename(sourcefn)
    texfn = sourcefn_base+'.tex'

    print "Processing %s" % sourcefn
    # load and parse source document
    f = open(sourcefn)
    xhtmltree = et.parse(f)
    f.close()

    # find all elements with attribute class='eq'
    eqs = [element for element in xhtmltree.getiterator()
           if element.get('class','')=='eq']
    # equations are now available in the eqs[..].text variable

    # create a LaTeX document and insert equations
    f = open(texfn,'w')
    f.write(tex_preamble)
    counter = 1
    for eq in eqs:
        if eq.tag == 'span': # inline equation
            f.write("$%s$ \n \\newpage \n" % eq.text)
        else:
            f.write("\\[\n%s \n\\] \n \\newpage \n" % eq.text)
        # delete LaTeX code from the document tree, and replace
        # them by image urls.
        latex_code = eq.text
        del eq.text
        imgname = "%seq%s%i.png" % (imgpath,sourcefn_base, counter)
        et.SubElement(eq,'img',src=imgname, alt=latex_code, title=latex_code)
        counter += 1
    # end LaTeX document
    f.write('\end{document}')
    f.close()

    # compile LaTeX document. A DVI file is created
    os.system('latex %s' % texfn)

    # Run dvipng on the generated DVI file. Use tight bounding box.
    # Magnification is set to 1200
    cmd = "dvipng -T tight -x 1200 -z 9 -bg transparent " \
    + "-o %seq%s%%d.png %s" % (imgpath , sourcefn_base, sourcefn_base)
    os.system(cmd)



    # Remove temporary files
    os.remove(sourcefn_base+'.tex')
    os.remove(sourcefn_base+'.log')
    os.remove(sourcefn_base+'.aux')
    os.remove(sourcefn_base+'.dvi')

    os.chdir(cwd)

    # Write processed source document to dest
    xhtmltree.write(destfn)

    print "Done."



if __name__=="__main__":

#    sourcefn = sys.argv[1]
#    destfn = sys.argv[2]
    sourcefn = "../docs/user_guide.html"
    destfn = "../docs/user_guide_eq.html"
    embed_latex_in_html(sourcefn, destfn)