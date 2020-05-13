#!/bin/bash
for f in `ls J*er_imp.kfx`;do
#for f in J26;do
	JXX=J${f:1:2}
	echo ${JXX}
	cp temp_er_imp.jpg.s0 ${JXX}_er_imp.jpg.s0
	sed -e "s/%JXX%/${JXX}/" temp_er_imp.jpg.s0 > ${JXX}_er_imp.jpg.s0
	kfxview ${JXX}_er_imp.jpg.s0
done


