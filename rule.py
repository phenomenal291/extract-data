import re

def phone(text):
    """
    Extract and normalize Vietnamese phone numbers from text.
    
    Args:
        text (str): Input text to search
        
    Returns:
        list: Normalized phone numbers starting with '0'
    """
    phone_regex = r'''(?:
                        (?:(?:(?:\+84|0084)|0)[\s.-]?(?:[13456789][\s.-]?(?:\d[\s.-]?){7}\d))
                        |
                        (?:(?:(?:02)(?:\d[\s.-]?){8,9}))
                        |
                        \((?:0\d{2,3})\)[\s.-]?(?:\d[\s.-]?){7}
                        |
                        (?:(?:1800|1900)[\s.-]?(?:\d[\s.-]?){4,6})
                        |
                        \((?:\+84|0084)\)[\s.-]?(?:\d[\s.-]?){8}\d
                        )'''

    found_numbers = re.findall(phone_regex, text, flags=re.VERBOSE)

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

def operating_license(text):
    """
    Extract and normalize Vietnamese operating license numbers from text.
    Args:
        text (str): Input text to search
        
    Returns:
        list: Normalized operating license numbers
    """
    # pattern: xxxx/yyy-GPHĐ 
    license_regex = r'\b\d{2,5}\s*/\s*(HCM|SYT)(?:-GPHĐ)?\b'
    
    full_licenses = []
    for match in re.finditer(license_regex, text):
        license = match.group(0)
        normalized = re.sub(r'\s*/\s*', '/', license)
        if not normalized.endswith('-GPHĐ'):
            normalized += '-GPHĐ'
        full_licenses.append(normalized)
    
    return full_licenses