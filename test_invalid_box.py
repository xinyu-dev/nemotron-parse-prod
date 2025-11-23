def check_invalid_bboxes(json_data):
    """
    Check if any bounding boxes in the extracted JSON data are invalid.
    
    A bbox is invalid if:
    - Any of xmin, ymin, xmax, ymax is negative
    - xmin >= xmax
    - ymin >= ymax
    
    Args:
        json_data: Dictionary containing the parsed JSON data or path to JSON file
        
    Returns:
        tuple: (is_valid, invalid_bboxes)
            - is_valid: Boolean indicating if all bboxes are valid
            - invalid_bboxes: List of dictionaries containing invalid bbox info
    """
    import json
    
    # If json_data is a string (file path), load the file
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            data = json.load(f)
    else:
        data = json_data
    
    invalid_bboxes = []
    
    # Iterate through all content items
    for item in data.get('content', []):
        extraction_id = item.get('extraction_id')
        metadata = item.get('metadata', {})
        bbox = metadata.get('bbox', {})
        
        # Get bbox coordinates
        xmin = bbox.get('xmin')
        ymin = bbox.get('ymin')
        xmax = bbox.get('xmax')
        ymax = bbox.get('ymax')
        
        # Skip if bbox is empty (some items might not have bbox)
        if xmin is None or ymin is None or xmax is None or ymax is None:
            continue
        
        # Check for invalid conditions
        issues = []
        
        if xmin < 0:
            issues.append(f"xmin ({xmin}) is negative")
        if ymin < 0:
            issues.append(f"ymin ({ymin}) is negative")
        if xmax < 0:
            issues.append(f"xmax ({xmax}) is negative")
        if ymax < 0:
            issues.append(f"ymax ({ymax}) is negative")
        if xmin >= xmax:
            issues.append(f"xmin ({xmin}) >= xmax ({xmax})")
        if ymin >= ymax:
            issues.append(f"ymin ({ymin}) >= ymax ({ymax})")
        
        # If there are issues, add to invalid list
        if issues:
            invalid_bboxes.append({
                'extraction_id': extraction_id,
                'type': metadata.get('type'),
                'bbox': bbox,
                'issues': issues
            })
    
    is_valid = len(invalid_bboxes) == 0
    
    return is_valid, invalid_bboxes


# Usage example
if __name__ == "__main__":
    # Test with the file
    file_path = "output_results/json_outputs/encoding_page_5_fitz_resized_extracted.json"
    
    is_valid, invalid_bboxes = check_invalid_bboxes(file_path)
    
    if is_valid:
        print("✓ All bounding boxes are valid!")
    else:
        print(f"✗ Found {len(invalid_bboxes)} invalid bounding box(es):\n")
        for invalid in invalid_bboxes:
            print(f"Extraction ID: {invalid['extraction_id']} ({invalid['type']})")
            print(f"  Bbox: {invalid['bbox']}")
            print(f"  Issues:")
            for issue in invalid['issues']:
                print(f"    - {issue}")
            print()