# <font color=steelblue>Linux Command Line & Shell Scripting Notes</font>
##### Author: <font color=salmon>Ankoor Bhagat</font></h5>

Linux Command Line 
--
- `$ pwd` - Print working directory
- `$ clear` - Clear terminal screen
- `$ ls` - List directory content in the current directory. `$ ls <file/directory>` - List files or directories. `$ ls -l` - List files or directories in long format. `$ ls -a` - List all files or directories (in a given directory). `$ ls -lS` - List files or directories sorted by size in long format. `$ ls -R` - List files or directories recursively. `$ ls -ld */` - List all directories in the current directory.

```shell
ls -l (output columns)
----------------------
<type>
<user-permissions>
<group-permissions>
<others-permissions> 
<symbolic-link> 
<user> 
<group> 
<file-size> 
<datetime> 
<name>
```


- `$ man <command>` - Manual for command 
- `$ cd <directory>` - Change directory to some directory. 
- `~` - Home directory
- `/` - Root directory
- `.` - Current directory
- `..` - Parent directory of current directory
- `$ cd` - Change directory to home. `$ cd ~` - Change to home directory. `$ cd /` - Change to root directory. `$ cd ..` - Change to parent directory of current directory.
- `$ cat <file-1> <file-2>...<file-n>` - Display text file(s). Do not use with large files.
- `$ cat -b <file-1> <file-2>...<file-n>` - Display text file(s) with line number added to non blank lines
- `$ cat -n <file-1> <file-2>...<file-n>` - Display text file(s) with line number added to all lines
- `$ cat -s <file-1> <file-2>...<file-n>` - Squeezes multiple blank lines and display text file(s) by adding a single blank line

- `<` - Redirects input to screen. `$ <command> < <filename.txt>` - Run `<command>` with input from `<filename.txt>`, instead of the keyboard, e.g. `$ sort < file.txt` . 
- `>` - Redirects output to a file instead of terminal screen, overwriting the file. Usage `$ <command> > <filename.txt>`. NOTE `1>` is STANDARD OUT and `2>` is STANDARD ERROR, e.g. `<command> 2> <filename.txt>` - Redirects standard error to a file.
- `>>` - Redirects output to a file appending the redirected output at the end.
- `<command-1> | <command-2>` - Pipe the output of `<command-1>` to the input of `<command-2>`
- `$ cat > <filename>.txt` - Create new text file. Then enter text on terminal and use `CTRL + d`. NOTE: This overwrites file if it already exists.
- `$ cat >> <filename>.txt` - Used for appending text to the existing file.
- `$ cat <file-1> <file-2>...<file-n> > <filename>.txt` - Concatenate multiple files and redirect output to a new file.
- `$ mkdir <directory-name>` - Create a new directory with name `<directory-name>`
- `$ mkdir -p <directory>/<sub-directory>` - Create a new directory and sub-directory
- `$ mkdir -p <directory>/{<sub-directory-1>,...,<sub-directory-n>}` - Create a new directory with `n` sub-directories. NOTE: There should be no **space** between commas.
- `$ rmdir <directory-1> <directory-2>...<directory-n>` - Removes empty directories.
- `$ rmdir -p <directory>/<sub-directory>` - Removes directory and sub-directories. Use `-v` option for verbose.
- `$ rm <file-1> <file-2>...<file-n>` - Removes files.
- `$ rm -rv <directory>/<sub-directory>` - Recursively removes directory and sub-directories when sub-directories contains files.
- `$ cp <from-filename> <to-filename>` - Copies content from `<from-filename>` to `<to-filename>` (creates if it does not exist)
- `$ cp <file-1> <file-2> ... <file-n> <directory>` - Copies files to a directory. This overwrites if file exists in directory, to avoid this use `-i` (interactive) option
- `$ cp -R <directory> <new-directory>` - Copies content of a directory recursively to another directory if the directory does not exists (it is created), if it exists the directory is copied into existing directory.
- `$ mv <old-filename> <new-filename>` - Renames file
- `$ mv <source-file/directory> <destination-file/directory>` - Moves source to destination. If file exists in the destination directory it will be overwritten, to avoid this use `-i` (interactive) option. If `<destination-file/directory>` does not exist then it will be created and contents of  `<source-file/directory>` will be moved. If `<destination-file/directory>` exists then `<source-file/directory>` will be moved.
- `$ mv <directory>/* .` - Move a directory's files one level up. 
- `$ less <filename>` - Display the content of the file. Use **up**, **down** keys to navigate up or down one line at a time and **space** key to navigate a page down at a time or **b** key to navigate a page up at a time. **shift + g** to end of the file. **1** and then **g** to the beginning of the file. Use `\<pattern>` to search for a pattern from top to bottom, use `n` to go to next highlighted pattern. Use `?<pattern>` to search for a pattern from bottom to top. Use `q` to quit. [`less` cheatsheet.](http://sheet.shiar.nl/less)
- `$ touch <filename>` - Creates new empty files. It is also used to change the timestamps on existing files by using `$ touch <existing-filename>` to current timestamp. NOTE: The file created by `touch` will have only `r` and `w` permissions but not `x` permissions so permissions need to be changed.
- `$ sudo <command>` - Allows a permitted user to execute a command as the superuser.
- `$ top` - Display and update sorted information about processes. `$ htop` - Display system stats more powerfully. Press `?` or `h` for help, and then press keys accordingly.
- `$ ps -ux` - Process status of all running processes under current user. `$ ps -aux` - Process status of all running processes under all users. `$ ps -U <username>` - Process status of all running processes under a particular user. `$ ps -C <process-name>` - Process status of all instances of a running processes.
- `$ pidof <process-name>` - Find the process ID of a running program.
- `$ kill <PID>` - Kill a process. `$ kill -KILL <PID>` - Forces the process to be killed. `$ kill -9 <PID>` - Used to kill process when they are not killed by the simple `kill` command. NOTE: If a process is killed and started again its PID will be changed. `$ kill -<signal> PID` - Send `<singal>` to process.
- `$ echo "<string>"` - Displays string on screen. Declaring a variable: `$ <variable-name>=<value>`, use `$ echo $<variable-name>` to display variable value. NOTE: There is no ` ` between `=` and `<variable-name>`. Display string with escape sequences on screen: `$ echo -e <escape-sequence>`, e.g. sequences `\n`, `\t`, `\c` - Suppress the trailing newline (i.e. to keep cursor on the same line after string is displayed on the scring). 
- `$ for f in ./**/*.txt; do tail -5 $f; done` NOTE: `**` does recursive search.
- `$ !!` - Runs the previous command
- **`ctrl + z`** - Stop (pause) a process, the output shows information (job number, job state, name of process) regarding paused process. 
	- `$ jobs` - Prints the list of jobs in this session.
	- `$ fg` - Brings the most recently stopped (paused) program to the foreground. Use `$ fg <job-#>` to bring some specific paused program to the foreground.
	- `$ <command> &` - To run a command as a background process.
- **`ctrl + c`** - Terminate (exit) a process.
- `$ chmod <u/g/o><+/-><r/w/x> <filename/directory>` - Change file or directory mode bits (permissions). `-R` flag for recursively changing directory permissions.
- `$ sudo chown <new-user> <file>` - Change the user and/or group ownership of each given file.

```shell
d|rwx|rwx|rwx --- <type>|<user>|<group>|<other>
- <type> --- Normal file
d <type> --- Directory

u - User
g - Group
o - Other
+/- - Adding or removing premissions
r - Read
w - Write
x - Execute

Examples of Symbolic Permissions
--------------------------------
chmod g+rw file 
chmod ug=rx file
chmod a+rwx file    # a -- all
chmod +x file    # x permission to all
chmod -w file    # Remove w permission for all
```
- `$ chmod <nnn> <filename/directory>`

```shell
Numeric/Octal Permission
------------------------
User    Group    Other
rwx     rwx      rwx
111     111      111    # Binary Notation
421     421      421    # Octal Notation
------------------------
4 + 2 + 1 = 7 ---> rwx (Binary: 111)
4 + 2     = 6 ---> rw  (Binary: 110)
4 + 1     = 5 ---> rx  (Binary: 101)
4             ---> r   (Binary: 100)
2 + 1     = 3 ---> wx  (Binary: 011)
2             ---> w   (Binary: 010)
1             ---> x   (Binary: 001)
0             ---> --- (Binary: 000)

Examples of Numeric/Octal Permissions
-------------------------------------
chmod 007 file    # No permission for user and group
chmod 764 file
```
- `chown -R <user-name>:<group> <directory>` - Change ownership of a directory recursively.
- `$ which <command>` -  Locates a program file in the user's path (`$PATH` environment variable)
- `$ whatis <command>` - Display one-line manual page description of a command.
- `.bashrc` file - It is a script that is executed whenever a new terminal session is started in interactive mode. This file can be used to customize commands, set environment variables, etc. 

```
# Alias of a command in .bashrc file
alias ls='ls --color=auto -l'
```
- `$ df -h` - Reports file system disk space usage in human readable format.
- `$ du -sh <directory>` - Estimates file space usage or directory space usage and reports summary in human readable format.
- `$ free` - Display amount of free and used memory in the system.
- `$ watch <command>` - Execute a program periodically, showing output fullscreen, e.g. `$ watch free -h`
- `$ head <file-1> <file-2>...<file-n>` - Output the first 10 lines of file(s). Use `-n <#-of-lines>` flag to change number of output lines. `-f` flag to display output appended data as the file grows.
- `$ tail <file-1> <file-2>...<file-n>` - Output the last 10 lines of file(s). `-n <#-of-lines>` flag to change number of output lines. `-f` flag to display output appended data as the file grows.
- `$ find <directory> -name <search-criteria>` - Search for files by *name* in a directory hierarchy. Placing quotes (`"`) around the search criteria avoids issues with wildcard characters. `-iname` flag - Find files by name (ignoring case). `find . type -f` flag - Find only files. `find . type -d` flag - Find only directories. `-size` flag - Find by size. `-mtime` flag - Find by modified time. [`find` primer](https://danielmiessler.com/study/find/)
- `$ wc <file-1> <file-2>...<file-n>` - Print newline, word, and byte counts for each input file. Output format: `#-lines #-words #-bytes filename`. `-w` flag for word count. `-l` flag for line count.
- `$ cal` - Display a calendar. `$ ncal` - Display a calendar with day information. `$ cal/ncal <YYYY>` - Displays a calendar for a year.
- `$ date` - Print or set the system date and time.
- `$ <command-1> ; <command-2> ; ... ; <command-n>` - Combine multiple commands and run them. NOTE: `;` runs all provided commands regardless of the success or failure of the previous command.
- `$ <command-1> && <command-2> && ... && <command-n>` - Combine multiple commands using `AND` and run them. NOTE: `&&` stops execution of next commands if previous command fails.
- `$ <command-1> || <command-2>` - `OR` operation. NOTE: If `<command-1>` fails then `<command-2>` will be executed else `<command-1>` is executed.
- `$ wget <URL>` - It is a package for retrieving files using HTTP, HTTPS, FTP and FTPS the most widely-used Internet protocols.
- `$ tar -cvf <archiv-name>.tar <directory>` - Archive a directory using `tar` archiving utility. `c` flag - Create new archive. `v` flag - Verbose. `f` flag - Following is the archive file name.
- `$ tar -xvf <archiv-name>.tar` - Extract files from archive.
- `$ grep "<pattern>" [flag] <file-1> <file-2> ... <file-k>` - Display lines matching a pattern. `i` flag - For ignoring case. `n` flag - Prefix each line of output with the 1-based  line number within its input file. NOTE: Use `*` wildcard for all files. Use `v` flag to display lines that do not match a pattern. [`grep` tutorial and primer](https://danielmiessler.com/study/grep/)
- `$ exit` - Cause normal process termination.
- `$ uptime` - Shows how long the system has been running.
- `$ diff <file-1> <file-2>` - Compares files line by line.
- `$ apropos <keywords-to-search>`- Search the manual page names and descriptions.
- `$ ping <ip-address>` - Sends ICMP ECHO_REQUEST to network hosts.
- `$ curl -O <URL>` - Download the file at the URL.

**Programs and etc.**

- `sort` - Sort the lines of standard input and sends it to standard output.
- `/dev/null` - A special file that will delete anything written into it.
- `scp` - Secure copy (remote file copy program)
- `rsync` - A remote (and local) file copying tool.
- `./configure` - Run the configure script (when inside the project directory) that comes with the source code. This creates a *Makefile*
- `make` - A program for building things (usually run from within the project directory).

**Package Handling**

- `$ apt-get update` - Resynchronize the package index files from their sources, i.e. update computer's catalog of available software. NOTE: An `update` should always be performed before an `upgrade`.
- `apt-cache search <PATTERN>` - Search the available packages for a pattern
- `$ apt-get install <package>` - Install a package using APT (Advanced Packaging Tool) package handling utility.
- `$ apt-get upgrade` - It is used to install the newest versions of all packages currently installed on the system from the sources enumerated in `/etc/apt/sources.list`. This is used to make sure that all the packages are upgraded with latest version of packages.
- `$ apt-get remove <package>` - Remove a package. Use `-purge` flag to remove all the configuration files related to a package.
- `$ apt-get autoremove` - Used to remove packages that were automatically installed to satisfy dependencies for other packages and are now no longer needed.
- *Steps to build software from source*: `$ sudo apt-get install build-essential` - Install the tools needed to build software from source code.
	1. Download source file
	2. Uncompress the source file
	3. Run `./configure` script
	4. Run `make` command
	5. Run `sudo make install` to install the software on computer


**Creating Users**

- `$ whoami` - Print the user name associated with the current effective user ID.
- `$ sudo adduser <user-name>` - Add user to the system.
- `$ su <user-name>` - Switch user ID or become superuser. NOTE: Without `<user-name>` `su` defaults to becoming the superuser. Use `$ exit` to revert back to previous user you were.


**Environment Variables** 

- Stores configuration information on computers. Environment Variables are written in upper case and the values they hold will be strings. Some of system environment variables are:
	- `HOME` - Path to home directory.
	- `PATH` - It is a list of directories separated by `:`. It represents the list of directories to search for when running an executable.
	- `PS1` - Defines format of command line prompt.

- `$ env` - Prints all of the environment variables.
- `$ <VARIABLE>=<value>` - Set a local environment variable.
- `$ export <VARIABLE>=<value>` - Set an environment variable that will be visible to child processes. Adding a directory (high priority search order) to `PATH` environment variable: `$ export PATH="</path/to/directory>:$PATH"` and directory (low priority search order) to `PATH` environment variable: `$ export PATH="$PATH:</path/to/directory>"`. NOTE: Usually environment variables are set in files such as `.bashrc` or `.profile` or `.zshrc`, etc. which runs automatically when new sessions are started.




**VIM Editor**

- [`vim` tutorial and primer](https://danielmiessler.com/study/vim/)


Bash (Bourne Again Shell) Scripting
--
- Shell types supported by an operating system: `$ cat /etc/shells`
- Location of bash: `$ which bash`
- Every shell script starts with `#!<location-of-bash>` in first line, e.g. `# /bin/bash`. NOTE: Need to give `x` permission to the script.

```shell
#!<bash-location>
<commands>
```

- To make bash script work in **zsh** use `#!/usr/bin/env bash` in first line.
- `# <comment>` - For writing comments in the script.
- Variables - Use **`$<variable>`** to access variable.
	- System Variables - Created and maintained by Linux OS, e.g. `BASH`, `HOME`, `PATH` etc., and are defined using Caplital letters.
	- User Defined Variables - Created and maintained by user. Define variable as follows: **`<variable>=value`**. NOTE: There is no space before and after `=`
- Reading inputs from terminal in a script: `read <variable-1> <variable-2>...<variable-n>` - Reads the inputs and assigns them to variables. The default variable for a single input case is `REPLY`
- `read -p "<string>" <variable>` - Allows to read input in the same line.
- `read -sp "<string>" <variable>` - Allows to read input in the same line without showing input on terminal, e.g. for *password*.
- `read -a <array>` - Allows to read multiple inputs and assign them to an array. Access the array elements using `${<array>[<index>]}`.
- Passing arguments to a bash script: `<script>.sh <arg-1> <arg-2>...<arg-n>`
	- Parsing arguments: `$1` is argument-1, `$2` is argument-2, ..., `$n` is $n^{th}$ argument, `$0` is `<script>.sh`
	- Parsing arguments into an array: `<array>=("$@")`, `$@` stores arguments as an array where argument-1 is stored at $0^{th}$ index of the array.
	- Number of arguments passed into the script: `$#`
	- Get process id of the script: `$$`
-  `exit` - Command terminates a script.
- `$ sleep <int/float>[suffix]` - Delay for a specified amount of time. Suffix `s` for
       seconds (the default), `m` for minutes,  `h`  for hours  or  `d` for days.
- **Wildcards**
	- `*` - Represents any number of characters (including zero), e.g. `ls *{<pattern-1>,<pattern-2>...<pattern-k>}` will match filenames containing these patterns.
	- `?` - Represents any single character.
	- `[]` - Specifies a range.
	- [Wildcards and Regular Expressions](http://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)

**Comparison Operators**

- Integer
	- `-eq` - is equal to, e.g. `if [ $a -eq $b ]`
	- `-ne` - is not equal to, e.g. `if [ $a -ne $b ]`
	- `-gt` - is greater than, e.g. `if [ $a -gt $b ]`
	- `-ge` - is greater than or equal to, e.g. `if [ $a -ge $b ]`
	- `-lt` - is less than, e.g. `if [ $a -lt $b ]`
	- `-le` - is less than or equal to, e.g. `if [ $a -le $b ]`
	- NOTE: If `<` or `<=` or `>` or `>=` are used then enclose the condition in `((` and `))`, e.g. `(($a > $b))`
- String
	- `=` or `==` - is equal to, e.g. `if [ $a = $b ]`
	- `!=` - is not equal to, e.g. `if [ $a != $b ]`
	- `-z` - String is null (length = 0), e.g. `if [ -z $a ]`

**File Test Operators**

- `-e` - Checks if file exists. e.g. `if [ -e $file ]`
- `-f` - Checks if file is an ordinary file as opposed to a directory or special file. e.g. `if [ -f $file ]`
- `-d` - Checks if file is a directory. e.g. `if [ -d $file ]`
- `-b` - Checks if file is a block special file (binary, image, audio, video). e.g. `if [ -b $file ]`
- `-c` - Checks if file is a character special file (normal file that contains text). e.g. `if [ -c $file ]`
- `-s` - Checks if file has size greater than 0 (i.e. file not empty or empty). e.g. `if [ -s $file ]`
- `-r` - Checks if file is readable. e.g. `if [ -r $file ]`
- `-w` - Checks if file is writable. e.g. `if [ -w $file ]`
- `-x` - Checks if file is executable. e.g. `if [ -x $file ]`

- **If Statement**
	- `[` is a symbolic link to `test`. NOTE: There needs to be a space between `[` and `<condition>` and `]`

```shell
if [ <condition> ]
then
  <if-statement>
fi
```
- **If-Else Statement**


```shell
if [ <condition> ]
then
  <if-statement>
else
  <else-statement>
fi
```
- **If-Elif-Else Statement**

```shell
if [ <condition-1> ]
then
  <if-statement>
elif [ <condition-2 ]
then
  <elif-statement>
else
  <else-statement>
fi
```

- **Logical `AND` Operator in If Statement** 
	- `if [ <condition-1> ] && [ <condition-2> ]`
	- `if [ <condition-1> -a <condition-2> ]` - `-a` is *AND* operator
	- `if [[ <condition-1> && <condition-2> ]]` - Need to use double square brackets 
- **Logical `OR` Operator in If Statement** 
	- `if [ <condition-1> ] || [ <condition-2> ]`
	- `if [ <condition-1> -o <condition-2> ]` - `-o` is *OR* operator
	- `if [[ <condition-1> || <condition-2> ]]` - Need to use double square brackets 
- **Arithmetic Operations (Integer)**
	- Common operators are `+`, `-`, `*`, `/`, `%`
	- The arithmetic operations should be enclosed in `((` and `))`. NOTE: There needs to be a space between `((` and `<arithmetic-operation>` and `))`, e.g. Addition: `$(( <variable-1> + <variable-2> ))`. 
	- Another way to perform arithmetic operations using `expr`, e.g. Multiplication: `$(expr <variable-1> \* <variable-2>)`. NOTE: Need to use `\*` for multiplication when using `expr` because the `*` character is not escaped.

- **Case Statement** - `case` is usually used to avoid nested `if` statements. NOTE: Each `<statement>` must be terminated with `;;`. `<pattern>` can be fixed or a regular expression. 
 
```shell
case <expression> in
  <pattern-1> )
    <statement-1> ;;
  <pattern-2> )
    <statement-2> ;;
  ...
  <pattern-k> )
    <statement-k> ;;
esac
```

- **Array Variables**
	- Declaring an array: `<array>=(<item-1> <item-2> ... <item-n>)`. NOTE: Space (` `) is used as a separator.
	- Print all elements of an array: `$ echo ${<array>[@]}`
	- Print indices of elements of an array: `$ echo ${!<array>[@]}`
	- Accessing array elements: `${<array>[<index>]}`
	- Length of an array: `${#<array>[@]}`
	- Adding an element to an array: `$ <array>[index]=<value>`
	- Removing an element located at some index in an array: `$ unset <array>[<index>]`.
	- If any variable (e.g. string) is treated as an array, the variable's value will be assigned to the array's $0^{th}$ index.

	
- **While Loop** - If `<condition>` is TRUE then only commands are executed.

```shell
while [ <condition> ]
do
  <command-1>
  <command-2>
  ...
  <command-n>
done
```

- **Until Loop** - If `<condition>` is FALSE then only commands are executed.

```shell
until [ <condition> ]
do
  <command-1>
  <command-2>
  ...
  <command-n>
done
```

- **For Loop**

```shell
## Simple 
for i in 1 2 3 4 5
do
  <command>
done
```

```shell
## Using range: Bash > 3.0 and with step: {<start>..<end>..<step>} - Bash > 4.0
for i in {<start>..<end>}
do
  <command>
done
```

```shell
## Using output of a command, e.g. ls 
for i in $(<command>)
do
  <some-command>
done
```

```shell
## Using expressions 
for i in (( i=start; i<end; i++ ))
do
  <some-command>
done
```

- **Select Loop** - Used to create menus and it is usually used with `case` statement.

```shell
select <variable> in <list>
do
  <command-1>
  <command-2>
  ...
  <command-n>
done
```

- **Break** - Used to exit or break out of the current loop prematurely.

```shell
for i in 1 2 3 4 5
do
  if [ <condition> ]
  then
    break
  fi
  <command>
done
```

- **Continue** - Used to skip the execution for that iteration.

```shell
for i in 1 2 3 4 5
do
  if [ <condition> ]
  then
    continue
  fi
  <command>
done
```

- **Functions** NOTE: If a function and argument is called in an `if` statement then use: `if ( <function-name> <argument> )`. Sometimes defining a function using keyword `function` does not work on **zsh** so add `#!/usr/bin/env bash` to the first line of the script. Use the `return` command to end the function, and return the supplied value to the calling section of the shell script.

```shell
# Notation-1
function <name>(){
  <commands>
  }
  
# Notation-2
<name>(){
  <commands>
  }
  
# Call the function without argument
<name> 

# Call the function with arguments
<name> <arguments>
```

- **Global & Local Variables** - All variables are `global` in shell script (even if the variable is defined in a function). To make a variable defined in a function to be a local variable to the function use: `local <variable>=<value>`

```shell
#! /bin/bash

function print() {
 local name=$1 # Remove local to see how all variables are global in a shell script.
 echo "The name is $name"
}

name="Ankoor"
echo "The name is $name [Before]"

print Crush
echo "The name is $name [After]"
```

- **Ternary Operation**

```shell
[ -f <file> ] && return 0 || return 1  # NOTE: Try [[ and ]] if there is error 

if [ -f <file> ]; then return 1; else return 0; fi
```

- **`readonly`** It is used to mark variables or functions (`-f` flag) read-only or unchangable, e.g. `readonly <variable>` or `readonly -f <function-name>`

- **Signals & Traps** - Signal is message sent to a process by the operating system. Example: If a script is running and `CTRL + C` is pressed, the script will be terminated. The script was in the middle of something, but the signal `CTRL + C` terminated it. NOTE: `$ man 7 signal` - Returns overview of signals. `0` is a success signal.
	- `CTRL + C` is called *interrupt signal* or *SIGINT*
	- `CTRL + D` is called *quit signal* or *SIGQUIT*
	- `CTRL + Z` is called *suspend signal* or *SIGTSTP*
	- `-9` is called *SIGKILL* used with `$ kill -9 <PID>`

There are some scenarios where the script is interrupted in the middle of doing something by some signal or some unexpected behavior. `trap` provides the script to trap or capture the *signals* or unexpected behavior within the script. `$ trap "<command>" <signal(s)-or-value(s)>` - Whenever `trap` receives `<signal-or-value>` it executes `<command>` inside `"`'s. NOTE: `trap` cannot capture `SIGKILL` or `SIGSTOP`

- **Debugging Script** 
	- `bash -x </path/to/script.sh>` - Prints verbose operations and executions of commands in the script. 
	- Another way is to use `#!/usr/bash -x` in the first line of script and call script normally (NOTE: This does not work with **`zsh`**).
	- Add `set -x` and `set +x` in the script. Debugging will start from the point where `set -x` is added and stops at point where `set +x` is added in the script.

- **Single Quotes: '** - Single quotes (`'`) are used to preserve the literal value of each character enclosed within the quotes. Single quotes prevent the variable expansion.

```shell
$ var=10
$ echo '$var'
> $var
```

- **Double Quotes: "** - Enclosing characters in double quotes (`"`) preserves the literal value of all characters within the quotes, with the exception of **$** and **`** which retain their special meaning within double quotes. Search exceptions regarding **\\**. NOTE: Variables are expanded when enclosed in double quotes.

```shell
$ var=10
$ echo "$var"
> 10
```
- **Back-Quotes or Back-Ticks** - When enclosed inside back-ticks, the shell interprets something to mean "the output of the command inside the back-ticks." This is referred to as command substitution, as the output of the command inside the back-ticks is substituted for the command itself. Command substitution allows the output of a command to replace the command itself. This is often used to assign the output of a command to a variable. NOTE: Another way is to enclose the command with `$(` and `)`, and pass it as an argument to another command.

```shell
$ var=`<command>`

# OR

$ var=$(<command>)
```

- **`rsync` For Loop (Include/Exclude)**

```
for dir in */ ; do
    echo $dir
    rsync -ar -RP --include="caffe_log.*" --exclude="*.json" \
    --exclude="snapshots" --exclude="*.prototxt" --exclude="*.caffemodel" 
    --exclude="*.h5" --exclude="*.lock" $dir /home/ankoor/DeepLearning/logs
done
```


- **References**
	- [The Bash Academy](https://guide.bash.academy)
	- [Bash Guide](http://mywiki.wooledge.org/BashGuide)


Terminal Shortcuts
--

- **`ctrl + a`** - Move to the beginning of the line.
- **`ctrl + e`** - Move to the end of the line.
- **`ctrl + u`** - Delete everything in the command before cursor.
- **`ctrl + k`** - Delete everything in the command after cursor.
- **`ctrl + l`** - Clear the screen.
- [iTerm shortcuts](http://www.ifdattic.com/iterm-shortcut-keys/)
- [iTerm cheatsheet](https://gist.github.com/squarism/ae3613daf5c01a98ba3a)

	


Miscellaneous
--
- [Transmit - macOS file transfer app](https://panic.com/transmit/)
- [iTerm2 Terminal](https://www.iterm2.com/)
- [Oh My ZSH Configuration Manager](http://ohmyz.sh)

```shell
# Set ZSH
---------
$ bash
$ cd .byobu
$ vi .tmux.conf # And add "set -g default-shell /usr/bin/zsh" 
# and "set -g default-command /usr/bin/zsh" without "
$ cd
# Install Oh My Zsh
$ vi .zshrc # Change theme to ZSH_THEME="ys"
$ exit  
```

- [Shell setup](https://www.bretfisher.com/shell/)

**GPU**

- NVIDIA System Management Interface program - `$ nvidia-smi`
- *GPU Temperature* - `$ nvidia-smi -q -d temperature`
- [Utility for stress-testing NVIDIA GPU accelerators](https://github.com/Microway/gpu-burn)

```
$ git clone https://github.com/Microway/gpu-burn
$ cd gpu-burn
$ make
$ cd
$ ./gpu_burn 120 # 120 seconds single precision test
$ ./gpu_burn -d $(( 60 * 30 )) # 30 minute double precision test
```

**Create Isolated Jupyter Python Kernel**

```
# Check where Jupyter is reading its configuration files
$ jupyter --paths
config:
    /home/<USER>/.jupyter
    /usr/etc/jupyter
    /usr/local/etc/jupyter
    /etc/jupyter
data:
    /home/<USER>/.local/share/jupyter
    /usr/local/share/jupyter
    /usr/share/jupyter
runtime:
    /run/user/1002/jupyter

# Find where virtualenv is storing <vritual environment>'s python
$ workon <virtual environment>
$ which python
/home/<USER>/.virtualenvs/<virtual environment/bin/python
$ deactivate

# Go to directory where Jupyter is storing its data (data output from jupyter --paths)
$ cd /home/<USER>/.local/share/jupyter/kernels
$ mkdir <virtual environment>
$ cd <virtal environment>
$ touch kernel.json
# Add the following to kernel.json. NOTE: Path should be virtual environments python path.
{
 "argv": [ "/home/<USER>/.virtualenvs/<virtual environment>/bin/python", 
           "-m", 
		   "ipykernel",
           "-f", 
		   "{connection_file}"],
 "display_name": "<virtual environment>",
 "language": "python"
}
```

**Jupyter Set `iopub` Rate Limit**

```
# Generate Jupyter Notebook Config file (if file does not exist)
$ jupyter notebook --generate-config 

# Edit Config File
$ vim ~/.jupyter/jupyter_notebook_config.py

# Set iopub_data_rate_limit
*.iopub_data_rate_limit=10000000000
```

**Set Jupyter on New Machine**

```
mkdir -p $HOME/.jupyter/custom/
touch $HOME/.jupyter/custom/custom.css
echo ".container { width:95% \!important;}" >> $HOME/.jupyter/custom/custom.css
echo "#login_widget:before {content: \"$(hostname)\"; \
font-size: 25px; text-transform: capitalize; \
margin-left: 5px;}" >> $HOME/.jupyter/custom/custom.css
```

**Byobu Not Working (Solution)**

```
$ ssh ankoor@<ip-address> -t bash
$ rm -r .byobu
$ byobu-disable
$ killall -u ankoor

$ ssh ankoor@<ip-address>
$ byobu-enable
```

**Git Project Contribution**

```
1. Fork a repository from a "upstream" repository. Forked repository is known as "origin"

2. Clone "origin" repository on laptop. This cloned repository is known as "local"
$ git clone git@github.com:<user>/<example>.git

3. Change to "local" repository's directory
$ cd <example>

4. Set Address for the "upstream" remote
$ git remote add upstream git@github.com:<maintainer>/<example>.git

5. Update "local" master (i.e. Pull in "upstream" changes)
$ git checkout master
$ git fetch upstream
$ git merge upstream/master
$ git push origin master

6. Development Work: Make changes
$ git checkout -b <branch>
$ git add <file-a> <file-b> ... <file-n>
$ git commit -m "Updated files"
$ git push origin <branch>

7. If "upstream" master moves ahead while code in <feature> branch is being developed
# Update <feature> branch: First make sure to commit code changes to <feature> branch
$ git checkout <feature>
$ git add <changed files> # Then `git stash` if needed
$ git commit -m "<message>"
# Now switch to master branch
$ git checkout master
$ git fetch upstream
$ git merge upstream/master
$ git push origin master
# Now switch to <feature> branch
$ git checkout <feature>
$ git merge master # Resolve conflicts
$ git push origin <feature>

8. Squash commits into one (Interactive Rebase)
$ git log --oneline origin/master..<feature> # List the commits on <branch> that has been made since the last commit on origin/master
> 6c34529 Second commit with a fix
> 889f452 First commit
$ git rebase -i origin/master # Ascending order
pick 889f452 First commit # Leave this alone
squash Second commit with a fix
$ git log --oneline origin/master..<feature>
$ git push -f origin <feature> # Force push

# Delete Local Branch
$ git branch -d <feature>

# Delete Remote Branch
$ git push origin --delete <feature>

# Decorated git lot
$ git log --oneline --decorate
$ git log --numstat --decorate

# Pull Request to DR08
$ git checkout <branch>
$ git fetch upstream
$ git rebase upstream/master

# After Rebasing
$ git push origin <feature>
# Error: Updates were rejected because the tip of your current branch is behind its remote counterpart.
error: failed to push some refs to 'git@github.com:<user>/<example>.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
$ git push -f origin <feature>

# Fetch an upstream PR
$ git fetch upstream pull/<id>/head:<feature> # <feature> branch name is given by you
$ git checkout <feature>
$ git push origin <feature>
```



