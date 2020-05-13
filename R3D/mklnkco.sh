#!/bin/bash
for l in J*_vis_*.r3d; do
  s=${l::7}
  echo ${s}
  ln -s ${l} ${s}_exit.r3d
  #echo "'"${s}"',"
done
