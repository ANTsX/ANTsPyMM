
####################
import glob as glob 
gg=glob.glob("*/*test*py")
gg=glob.glob("tests/*py")
gg=glob.glob("docs/*localint.py")
gg.sort()
print( gg )
###
print("╱╲═╳╱╲◆╱╲═╳╱╲◆╱╲═╳╱╲=╱╲═╳╱╲◆╱╲═╳╱╲◆╱╲═╳╱╲")
ct=0
for fn in gg:
	print(fn)
	if ct >= 0:
		exec(open(fn).read())
	print("╱╲═╳╱╲◆╱╲═╳╱╲◆╱╲═╳╱╲=╱╲═╳╱╲◆╱╲═╳╱╲◆╱╲═╳╱╲")
	ct=ct+1
