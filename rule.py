import re

def phone(text):
    """
    Extract and normalize Vietnamese phone numbers from text.
    
    Args:
        text (str): Input text to search
        
    Returns:
        list: Normalized phone numbers starting with '0'
    """
    phone_regex = r'(?:(?:(?:(?:\+84|0084)|0)[\s.-]?(?:\d[\s.-]?){9,10})|\((?:0\d{2,3})\)[\s.-]?(?:\d[\s.-]?){7})'

    found_numbers = re.findall(phone_regex, text)

    phone_lists=[]
    for number in found_numbers:
        clean_num=re.sub(r'[^\d]', '', number)
        
        if clean_num.startswith('84'):
            clean_num = '0' + clean_num[2:]
        if clean_num.startswith('0084'):
            clean_num = '0' + clean_num[4:]
        
        if clean_num:
            phone_lists.append(clean_num)
    return phone_lists