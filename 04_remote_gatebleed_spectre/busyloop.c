int main() {
	asm volatile("movl $0, %eax\n"
				 "movl $1, %ebx\n"
				 "movl $2, %ecx\n"
				 "movl $3, %edx\n"
				 "start:\n"
				 "add %eax, %ebx\n"
				 "add %ebx, %ecx\n"
				 "add %ecx, %edx\n"
				 "add %edx, %eax\n"
				 "jmp start\n");
}
