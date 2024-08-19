compiler  	=g++
nvcc		=clang++

cflags    	+=-std=c++20 -O3 -march=native -fopt-info-vec -fopt-info-inline  
linkflags 	+=-ldl -lcudadevrt  -lcudart_static  -lrt -lpthread 
nvflags		+=-std=c++20   -O3  --cuda-gpu-arch=sm_75 -w 


SrcDir	  		=./src
ObjDir    		=./obj
HeadRoot		=./include
HeadDir         +=$(HeadRoot) $(foreach dir,$(HeadRoot),$(wildcard $(dir)/*))
source    		=$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp $(dir)/*.cu))
head      		=$(foreach dir,$(HeadDir),$(wildcard $(dir)/*.hpp $(dir)/*.cuh))
object    		=$(patsubst %.cpp,$(ObjDir)/%.o,$(patsubst %.cu,$(ObjDir)/%.cu.o,$(notdir $(source))))

target 	  		=ScriptMemory
NO_COLOR		=\033[0m
OK_COLOR		=\033[32;01m

$(target):$(object) $(head)
	$(nvcc) -o $(target) $(object) $(linkflags) 
	@printf "$(OK_COLOR)Compiling Is Successful!\nExecutable File: $(target) $(NO_COLOR)\n"

$(ObjDir)/%.o:$(SrcDir)/%.cpp $(head)
	$(compiler) -c $(cflags) $< -o $@ -I $(HeadRoot)

$(ObjDir)/%.cu.o:$(SrcDir)/%.cu $(head)
	$(nvcc) -c $(nvflags) $< -o $@ -I $(HeadRoot)

.PHONY:run 
run:$(target)
	@printf "$(OK_COLOR)$(target) is executing $(NO_COLOR)\n"
	./$(target)

.PHONY:clean	 
clean:
	-rm $(object) $(target)

.PHONY:clean_all	 
clean_all:
	-rm $(object) $(target) ./output/*

.PHONY:plot
plot:
	python3 Plot.py