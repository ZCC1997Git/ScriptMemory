compiler  	=g++
nvcc		=clang++

cflags    	+=-std=c++20 -O3 -march=native 
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

ALL:ScriptMemory_CUDA ScriptMemory_CPU

ScriptMemory_CUDA :./obj/main_cu.cu.o $(head)
	$(nvcc) -o $@ $< $(linkflags) 
	@printf "$(OK_COLOR)Compiling Is Successful!\nExecutable File: ScriptMemory_CUDA  $(NO_COLOR)\n"

ScriptMemory_CPU :./obj/main_cpu.o $(head)
	$(compiler) -o $@ $< -fopenmp
	@printf "$(OK_COLOR)Compiling Is Successful!\nExecutable File:ScriptMemory_CPU $(NO_COLOR)\n"

$(ObjDir)/%.o:$(SrcDir)/%.cpp $(head)
	$(compiler) -c $(cflags) -fopenmp $< -o $@ -I $(HeadRoot)

$(ObjDir)/%.cu.o:$(SrcDir)/%.cu $(head)
	$(nvcc) -c $(nvflags) $< -o $@ -I $(HeadRoot)

.PHONY:run_cuda
run_cuda:ScriptMemory_CUDA
	@printf "$(OK_COLOR)$(target) is executing $(NO_COLOR)\n"
	./ScriptMemory_CUDA

.PHONY:run_cpu
run_cpu:ScriptMemory_CPU
	@printf "$(OK_COLOR)$(target) is executing $(NO_COLOR)\n"
	./ScriptMemory_CPU

.PHONY:clean	 
clean:
	-rm $(object) ScriptMemory_CUDA ScriptMemory_CPU

.PHONY:clean_all	 
clean_all:
	-rm $(object) ScriptMemory_CUDA ScriptMemory_CPU ./output/*

.PHONY:plot
plot:
	python3 Plot.py