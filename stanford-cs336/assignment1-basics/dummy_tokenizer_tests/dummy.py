
char_obj = 'a'

# __repr__() output
print(repr(char_obj))

# Printed representation (using print())
print(char_obj)

print(chr(0))

print("this is a test" + chr(0) + "string")

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])




if __name__ == "__main__":
    print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
    # print(decode_utf8_bytes_to_str_wrong("café".encode("utf-8")))
    test_str = "hello".encode('utf-8')
    print(test_str)
    token_tuple = tuple(bytes([b]) for b in test_str)
