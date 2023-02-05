class DTree(object):
    
    def add_attribute(self, attribute) -> None:
        self.attributes.append(attribute)
    
    def __init__(self, node = {}, attributes = None, isLeaf = False) -> None:
        self.node = node
        self.isLeaf = isLeaf
        self.attributes = []
        
        if attributes is not None:
            for attribute in attributes:
                self.attributes.append(attribute)


