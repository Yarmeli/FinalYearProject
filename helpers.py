# Global variable to print debug messages
DebugMode = True
  
def Debug(area, message, printmsg = DebugMode):
    if printmsg:
        print(f"[Debug] [{area}] {message}")
        
def DebugSeparator(symbol="-", amount = 70, printmsg = DebugMode):
    if printmsg:
        print("[Debug] " + f"{symbol}" * amount)
   