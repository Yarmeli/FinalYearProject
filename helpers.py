# Global variable to print debug messages
DebugMode = True
  
def Debug(area, message, printmsg = DebugMode):
    if printmsg:
        print(f"[Debug] [{area}] {message}")
   