import React, { useState } from 'react'
import { 
  Box, 
  Typography, 
  Card, 
  CardContent,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Divider,
  TextField,
  IconButton,
  Button,
  Grid,
  Chip
} from '@mui/material'
import { 
  Email, 
  Sms, 
  Phone, 
  Twitter, 
  Send, 
  AttachFile,
  Search,
  FilterList,
  Refresh,
  MoreVert,
  Done,
  DoneAll,
  AccessTime,
  Error
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const Communications = () => {
  const [activeTab, setActiveTab] = useState(0)
  const [newMessage, setNewMessage] = useState('')
  const [selectedConversation, setSelectedConversation] = useState(null)

  const channels = [
    { name: 'Email', icon: Email, count: 12 },
    { name: 'SMS', icon: Sms, count: 5 },
    { name: 'Phone', icon: Phone, count: 0 },
    { name: 'Social', icon: Twitter, count: 8 }
  ]

  const conversations = {
    'Email': [
      { 
        id: 'email1',
        contact: 'John Smith',
        email: 'john.smith@company.com',
        preview: 'RE: Quarterly Report Analysis',
        time: '10:42 AM',
        unread: true,
        messages: [
          { sender: 'them', content: 'Hi, I was wondering if you had a chance to look at the quarterly report analysis?', time: '10:30 AM' },
          { sender: 'agent', content: 'Yes, I\'ve analyzed the report and found several key insights. The Q4 performance exceeded expectations by 12%, with particularly strong results in the Asia-Pacific region.', time: '10:35 AM' },
          { sender: 'them', content: 'That\'s great news! Could you prepare a presentation summarizing these findings for the board meeting next week?', time: '10:42 AM' },
        ]
      },
      { 
        id: 'email2',
        contact: 'Sarah Johnson',
        email: 'sarah.j@techinnovate.co',
        preview: 'New project proposal',
        time: 'Yesterday',
        unread: false,
        messages: [
          { sender: 'them', content: 'I\'m sending over the new project proposal for your review.', time: 'Yesterday, 2:15 PM' },
          { sender: 'agent', content: 'Thank you for sending this. I\'ll review it immediately and provide feedback.', time: 'Yesterday, 2:20 PM' },
          { sender: 'agent', content: 'I\'ve completed my review. The proposal looks strong, but I\'ve identified a few areas where we could strengthen the value proposition. I\'ve attached my detailed feedback.', time: 'Yesterday, 4:05 PM' },
        ]
      },
      { 
        id: 'email3',
        contact: 'Marketing Team',
        email: 'marketing@company.com',
        preview: 'Campaign performance metrics',
        time: '2 days ago',
        unread: false,
        messages: [
          { sender: 'them', content: 'Here are the latest campaign performance metrics for review.', time: '2 days ago, 11:00 AM' },
          { sender: 'agent', content: 'Thank you for sharing these metrics. I notice the conversion rates have improved by 15% since our last campaign. The social media engagement is particularly strong.', time: '2 days ago, 11:30 AM' },
        ]
      }
    ],
    'SMS': [
      { 
        id: 'sms1',
        contact: 'Alex Wong',
        phone: '+1-555-123-4567',
        preview: 'When can we schedule the client call?',
        time: '1 hour ago',
        unread: true,
        messages: [
          { sender: 'them', content: 'Hey, just checking on the client call time.', time: '1 hour ago, 11:15 AM' },
          { sender: 'them', content: 'When can we schedule it?', time: '1 hour ago, 11:16 AM' },
        ]
      },
      { 
        id: 'sms2',
        contact: 'Maria Garcia',
        phone: '+1-555-987-6543',
        preview: 'The files have been uploaded',
        time: 'Yesterday',
        unread: false,
        messages: [
          { sender: 'them', content: 'Just letting you know that all the project files have been uploaded to the shared drive.', time: 'Yesterday, 4:30 PM' },
          { sender: 'agent', content: 'Great, thank you for confirming. I\'ll start processing them right away.', time: 'Yesterday, 4:35 PM' },
        ]
      }
    ],
    'Social': [
      { 
        id: 'social1',
        contact: '@TechInnovations',
        platform: 'Twitter',
        preview: 'We loved your latest article on AI trends!',
        time: '3 hours ago',
        unread: true,
        messages: [
          { sender: 'them', content: '@EBAIFAgent We loved your latest article on AI trends! Would love to collaborate on future content.', time: '3 hours ago, 9:20 AM' },
        ]
      },
      { 
        id: 'social2',
        contact: '@DataScienceDaily',
        platform: 'Twitter',
        preview: 'What\'s your take on the latest GPT architecture?',
        time: 'Yesterday',
        unread: false,
        messages: [
          { sender: 'them', content: '@EBAIFAgent What\'s your take on the latest GPT architecture? Do you think it represents a significant advancement?', time: 'Yesterday, 3:45 PM' },
          { sender: 'agent', content: '@DataScienceDaily Absolutely! The improvements in context handling and specialized domain knowledge are substantial. We\'re particularly impressed with the efficiency gains while maintaining quality.', time: 'Yesterday, 4:00 PM' },
        ]
      }
    ]
  }

  const handleSendMessage = () => {
    if (!newMessage.trim() || !selectedConversation) return;
    
    // In a real app, this would send the message to the backend
    console.log(`Sending message to ${selectedConversation.contact}: ${newMessage}`);
    
    // Add message to conversation
    const updatedMessage = {
      sender: 'agent',
      content: newMessage,
      time: 'Just now'
    };
    
    // This would be done properly with state management in a real app
    selectedConversation.messages.push(updatedMessage);
    
    setNewMessage('');
  }

  const getChannelData = () => {
    return conversations[channels[activeTab].name] || [];
  }

  const handleConversationSelect = (conversation) => {
    setSelectedConversation(conversation);
  }

  return (
    <Box sx={{ p: 3, height: 'calc(100vh - 64px)' }}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h3" sx={{ color: 'white', mb: 4, fontWeight: 700 }}>
          💬 Communication Center
        </Typography>
      </motion.div>

      <Grid container spacing={3} sx={{ height: 'calc(100% - 80px)' }}>
        {/* Left panel - Conversations list */}
        <Grid item xs={12} md={4} lg={3}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TextField
                    fullWidth
                    placeholder="Search communications..."
                    size="small"
                    InputProps={{
                      startAdornment: <Search sx={{ color: '#b8c5d6', mr: 1 }} />,
                    }}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '12px',
                        background: 'rgba(255, 255, 255, 0.05)'
                      }
                    }}
                  />
                  <IconButton size="small" sx={{ ml: 1, color: '#b8c5d6' }}>
                    <FilterList />
                  </IconButton>
                </Box>

                <Tabs
                  value={activeTab}
                  onChange={(e, newValue) => {
                    setActiveTab(newValue);
                    setSelectedConversation(null);
                  }}
                  variant="fullWidth"
                  sx={{
                    '& .MuiTab-root': {
                      minWidth: 'auto',
                      color: '#b8c5d6',
                      textTransform: 'none',
                      fontSize: '0.9rem'
                    },
                    '& .Mui-selected': {
                      color: '#667eea'
                    },
                    '& .MuiTabs-indicator': {
                      backgroundColor: '#667eea'
                    }
                  }}
                >
                  {channels.map((channel, idx) => (
                    <Tab 
                      key={channel.name}
                      icon={<channel.icon sx={{ fontSize: 20 }} />}
                      label={channel.count > 0 ? `(${channel.count})` : ''}
                      iconPosition="start"
                    />
                  ))}
                </Tabs>
              </Box>

              <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

              <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                <List sx={{ p: 0 }}>
                  {getChannelData().map((conversation, idx) => (
                    <React.Fragment key={conversation.id}>
                      <ListItem 
                        button
                        selected={selectedConversation?.id === conversation.id}
                        onClick={() => handleConversationSelect(conversation)}
                        sx={{ 
                          py: 2,
                          background: selectedConversation?.id === conversation.id 
                            ? 'rgba(102, 126, 234, 0.1)' 
                            : 'transparent',
                          borderLeft: selectedConversation?.id === conversation.id 
                            ? '3px solid #667eea' 
                            : '3px solid transparent'
                        }}
                      >
                        <ListItemAvatar>
                          <Avatar 
                            sx={{ 
                              bgcolor: conversation.unread 
                                ? 'rgba(102, 126, 234, 0.2)' 
                                : 'rgba(255, 255, 255, 0.1)'
                            }}
                          >
                            {channels[activeTab].icon && <channels[activeTab].icon />}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                              <Typography 
                                variant="body1" 
                                sx={{ 
                                  color: 'white',
                                  fontWeight: conversation.unread ? 600 : 400
                                }}
                              >
                                {conversation.contact}
                              </Typography>
                              <Typography variant="caption" sx={{ color: '#b8c5d6' }}>
                                {conversation.time}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Typography 
                                variant="body2" 
                                sx={{ 
                                  color: conversation.unread ? 'white' : '#b8c5d6',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  fontWeight: conversation.unread ? 500 : 400,
                                  maxWidth: '200px'
                                }}
                              >
                                {conversation.preview}
                              </Typography>
                              {conversation.unread && (
                                <Box
                                  sx={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    bgcolor: '#667eea',
                                    ml: 1
                                  }}
                                />
                              )}
                            </Box>
                          }
                        />
                      </ListItem>
                      {idx < getChannelData().length - 1 && (
                        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.05)' }} />
                      )}
                    </React.Fragment>
                  ))}
                </List>
              </Box>

              <Box sx={{ p: 2 }}>
                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<channels[activeTab].icon />}
                  sx={{
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    borderRadius: '12px',
                    textTransform: 'none'
                  }}
                >
                  New {channels[activeTab].name}
                </Button>
              </Box>
            </Card>
          </motion.div>
        </Grid>

        {/* Right panel - Conversation */}
        <Grid item xs={12} md={8} lg={9}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {selectedConversation ? (
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                {/* Conversation header */}
                <Box sx={{ p: 2, display: 'flex', alignItems: 'center', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                  <Avatar sx={{ bgcolor: 'rgba(102, 126, 234, 0.2)', mr: 2 }}>
                    {channels[activeTab].icon && <channels[activeTab].icon />}
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                      {selectedConversation.contact}
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                      {selectedConversation.email || selectedConversation.phone || selectedConversation.platform}
                    </Typography>
                  </Box>
                  <Box>
                    <IconButton sx={{ color: '#b8c5d6' }}>
                      <Refresh />
                    </IconButton>
                    <IconButton sx={{ color: '#b8c5d6' }}>
                      <MoreVert />
                    </IconButton>
                  </Box>
                </Box>

                {/* Messages */}
                <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {selectedConversation.messages.map((message, idx) => (
                    <Box
                      key={idx}
                      sx={{
                        display: 'flex',
                        justifyContent: message.sender === 'agent' ? 'flex-end' : 'flex-start'
                      }}
                    >
                      <Box
                        sx={{
                          maxWidth: '70%',
                          p: 2,
                          borderRadius: '12px',
                          background: message.sender === 'agent' 
                            ? 'linear-gradient(135deg, #667eea80 0%, #764ba280 100%)' 
                            : 'rgba(255, 255, 255, 0.1)',
                          boxShadow: message.sender === 'agent'
                            ? '0 4px 12px rgba(102, 126, 234, 0.2)'
                            : 'none'
                        }}
                      >
                        <Typography variant="body1" sx={{ color: 'white', mb: 1 }}>
                          {message.content}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                          <Typography variant="caption" sx={{ color: '#b8c5d6', mr: 1 }}>
                            {message.time}
                          </Typography>
                          {message.sender === 'agent' && (
                            <DoneAll sx={{ fontSize: 16, color: '#4ecdc4' }} />
                          )}
                        </Box>
                      </Box>
                    </Box>
                  ))}
                </Box>

                {/* Message input */}
                <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <TextField
                      fullWidth
                      placeholder={`Type a message...`}
                      variant="outlined"
                      multiline
                      maxRows={4}
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          borderRadius: '12px',
                          background: 'rgba(255, 255, 255, 0.05)'
                        }
                      }}
                    />
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <IconButton sx={{ color: '#b8c5d6' }}>
                        <AttachFile />
                      </IconButton>
                      <IconButton 
                        sx={{
                          color: 'white',
                          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          '&:hover': {
                            background: 'linear-gradient(135deg, #7687ea 0%, #865cb2 100%)'
                          }
                        }}
                        onClick={handleSendMessage}
                        disabled={!newMessage.trim()}
                      >
                        <Send />
                      </IconButton>
                    </Box>
                  </Box>
                  <Typography variant="caption" sx={{ color: '#b8c5d6', mt: 1, display: 'block' }}>
                    AI Assistant is drafting responses in real-time with 99.8% accuracy
                  </Typography>
                </Box>
              </Card>
            ) : (
              <Card sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Box sx={{ textAlign: 'center', p: 4 }}>
                  <Avatar sx={{ width: 80, height: 80, margin: '0 auto 16px', bgcolor: 'rgba(102, 126, 234, 0.2)' }}>
                    <channels[activeTab].icon sx={{ fontSize: 40, color: '#667eea' }} />
                  </Avatar>
                  <Typography variant="h5" sx={{ color: 'white', mb: 2, fontWeight: 600 }}>
                    Select a {channels[activeTab].name} Conversation
                  </Typography>
                  <Typography variant="body1" sx={{ color: '#b8c5d6', mb: 3, maxWidth: '500px' }}>
                    The AI assistant can handle communications across multiple channels, adapting tone and content to each platform while maintaining context awareness.
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<channels[activeTab].icon />}
                    sx={{
                      borderColor: '#667eea',
                      color: '#667eea',
                      borderRadius: '12px',
                      textTransform: 'none',
                      '&:hover': {
                        borderColor: '#764ba2',
                        background: 'rgba(102, 126, 234, 0.05)'
                      }
                    }}
                  >
                    Create New {channels[activeTab].name}
                  </Button>
                </Box>
              </Card>
            )}
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Communications