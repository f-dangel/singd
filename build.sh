# write out (tangle) files
emacs README.org --batch -f org-babel-tangle --kill

# fix TABs, that were exported as spaces, in makefile
sed -i "s/^  /\t/g" makefile
