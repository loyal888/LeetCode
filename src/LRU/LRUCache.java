package LRU;

import java.util.HashMap;
import java.util.Map;

class ListNode {
    public Integer key;
    public  Integer value;
    public ListNode pre;
    public ListNode next;

    public ListNode() {
    }

    public ListNode(int key, int value) {
        this.key = key;
        this.value = value;
    }
}

/**
 * LRU 例子
 */
public class LRUCache {
    // 初始容量
    int capacity = 5;
    // hashmap 保存结点 保证O(1)复杂度
    Map<Integer, ListNode> hashmap = new HashMap<>();
    // 双向队列模拟插入删除
    ListNode head = new ListNode();
    ListNode tail = new ListNode();

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head.next = tail;
        tail.pre = head;
    }

    public void moveNodeToTail(int key) {
        // 取出hashmap 并移动到队尾
        ListNode node = hashmap.get(key);
        node.pre.next = node.next;
        node.next.pre = node.pre;

        node.pre = tail.pre;
        tail.pre.next = node;
        node.next = tail;
        tail.pre = node;
    }

    public Integer get(Integer key) {
        // 链表中已经有这个值，将其移动到队尾，变为最新访问的
        if (hashmap.containsKey(key)) {
            moveNodeToTail(key);
        }
        ListNode res = hashmap.getOrDefault(key, null);
        if (res == null) {
            return -1;
        }
        return res.value;
    }

    public void put(Integer key, Integer value) {
        // 如果key本身已经在哈希表中了就不需要在链表中加入新的节点
        //但是需要更新字典该值对应节点的value
        if (hashmap.containsKey(key)) {
            hashmap.get(key).value = value;
            // 之后将该节点移到末尾
            moveNodeToTail(key);
        } else {
            if (hashmap.size() == capacity) {
                // 去掉哈希表对应项
                hashmap.remove(head.next.key);
                head.next = head.next.next;
                head.next.pre = head;
            }
            //如果不在的话就插入到尾节点前
            ListNode newNode = new ListNode(key, value);
            hashmap.put(key,newNode);
            newNode.pre = tail.pre;
            newNode.next = tail;
            tail.pre.next = newNode;
            tail.pre = newNode;
        }
    }
}

