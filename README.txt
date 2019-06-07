Experiment planning tool for elastic neutron diffraction single-crystal experiments.

To create RPM:
change release number in setup.cfg
export PYTHONPATH=$PYTHONPATH:~/CrystalPlan/lib/python/
python setup.py install --home=~/CrystalPlan
python setup.py bdist  #generate RPM

ask linux-support to install on all analysis and instrument computers:
~/CrystalPlan/dist/CrystalPlan-1.2-RELEASENO.noarch.rpm           
