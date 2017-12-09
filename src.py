class Frame:
    counter = 0
    def __init__(self, name):
        self.id = Frame.counter+1
        self.f_name = name
        self.slots = []
        self.addSlot(['f_name',[name]])
        Frame.counter += 1

    def addSlot(self, slot):
        self.slots.append(slot)

    def listProp(self,props):
        parents = []
        for slot in self.slots:
            if(slot[0] == 'a_part_of' or slot[0] == 'is_a' or slot[0] == 'ako'):
                parents.append(slot[1][0])
            else:
                if(slot[0] in props):
                    continue
                print(slot[0], '(', self.f_name, "): ", end="")
                props.add(slot[0])
                for i,facet in enumerate(slot[1]):
                    if(i==len(slot[1])-1):
                        print(facet)
                    else:
                        print(facet,' ', end="")
        for parent in parents:
            parent.listProp(props)


if __name__ == '__main__':
    Frames = {}
    #Frame name has to be a string
    #A slot is of the form [ , [ , .., ]]
    #Slot name i.e. slot[0] has to be a string

    f1 = Frame('university')
    f1.addSlot(['phone',['default',11686971]])
    Frames[f1.f_name] = f1

    f2 = Frame('department')
    f2.addSlot(['a_part_of',[f1]])
    f2.addSlot(['programme',['BTech','MTech','PhD']])
    Frames[f2.f_name] = f2

    f3 = Frame('faculty')
    f3.addSlot(['a_part_of',[f2]])
    f3.addSlot(['age',['range','25 - 60']])
    Frames[f3.f_name] = f3
    f3.listProp(set([]))